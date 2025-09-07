from collections.abc import Callable, Sequence
import csv
import dataclasses
import datetime
import functools
import multiprocessing
from pathlib import Path
import matplotlib.pyplot as plt
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=true"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_CLIENT_MEM_FRACTION"] = "0.95"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform" # cudaMalloc/free
import pathlib
import shutil
import string
import textwrap
import time
import typing
from typing import overload
from Bio import PDB
from absl import app
from absl import flags
from alphafold3.common import folding_input
from alphafold3.common import resources
from alphafold3.constants import chemical_components
import alphafold3.cpp
from alphafold3.data import featurisation
from alphafold3.data import pipeline
from alphafold3.jax.attention import attention
from alphafold3.model import feat_batch
from alphafold3.constants import residue_names, atom_types
from alphafold3.model import features
from alphafold3.model import model
from alphafold3.model.model import get_predicted_structure
from alphafold3.model import params
from alphafold3.model import post_processing
from alphafold3.model.components import utils
import sys
import subprocess
from alphafold3.model.atom_layout.atom_layout import AtomLayout
import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np
import optax
from jax import remat
from jax import lax
import jax.random as jr
import math
from functools import partial
import dataclasses as dc


class Opt_af3deisgn:
   _BUCKETS = ['256', '512', '768', '1024', '1280', '1536', '2048', '2560', '3072','3584', '4096', '4608', '5120']
   _GPU_DEVICE = 0
   _FLASH_ATTENTION_IMPLEMENTATION = 'cudnn'
   _NUM_DIFFUSION_SAMPLES = 1
   _NUM_RECYCLES = 1
   _SAVE_EMBEDDINGS = True
   MODEL_DIR = "/data/share/alphafold3"
   _OUTPUT_DIR = '/home/ge/app/af3design/test'
   name = 'test'
   binder_length = 130
   target = ['RET'] # only ligand
   target_length = 21
   motif_pos = 120 # less than binder_length
   motif_pos_aa = 'K' # only one aa
   af3_bonds = [[["A",120,"NZ"],["B",1,"C15"]]] # "bondedAtomPairs": [[["A",231,"NZ"],["B",1,"C15"]]]
   stage = 'logits' # or 'logits'

opt_af3deisgn = Opt_af3deisgn()
opt_af3deisgn._BUCKETS = sorted([*map(int, opt_af3deisgn._BUCKETS), int(opt_af3deisgn.binder_length) + int(opt_af3deisgn.target_length)])
opt_af3deisgn._BUCKETS = list(map(str, opt_af3deisgn._BUCKETS))


def make_model_config(
    *,
    flash_attention_implementation: attention.Implementation = 'triton',
    num_diffusion_samples: int = 5,
    num_recycles: int = 10,
    return_embeddings: bool = False,
) -> model.Model.Config:
  """Returns a model config with some defaults overridden."""
  config = model.Model.Config()
  config.global_config.flash_attention_implementation = (
      flash_attention_implementation
  )
  config.heads.diffusion.eval.num_samples = num_diffusion_samples
  config.num_recycles = num_recycles
  config.return_embeddings = return_embeddings
  return config

def cif2pdb(cif_file_path, pdb_file_path):
    if not os.path.exists(cif_file_path):
        print(f'Error: {cif_file_path} does not exist')
        sys.exit(1)
    if os.path.exists(pdb_file_path):
        os.system(f'rm {pdb_file_path}')
    parser = PDB.MMCIFParser()
    structure = parser.get_structure('hh',  cif_file_path)
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(pdb_file_path)

def _get_helix_loss(
    dgram: jnp.ndarray,
    dgram_bins: jnp.ndarray,
    *,
    offset: jnp.ndarray | None = None,   # (L, L) 或 None
    mask_2d: jnp.ndarray | None = None,  # (L, L) bool/float
    binary: bool = False,
    **kwargs,
) -> jnp.ndarray:

    x = _get_con_loss(dgram, dgram_bins, cutoff=6.0, binary=binary)
    if offset is None:
        if mask_2d is None:
            return jnp.mean(jnp.diagonal(x, offset=3, axis1=-2, axis2=-1))

        mask_2d = mask_2d.astype(x.dtype)
        num = jnp.sum(jnp.diagonal(x * mask_2d, offset=3, axis1=-2, axis2=-1))
        den = jnp.sum(jnp.diagonal(mask_2d,     offset=3, axis1=-2, axis2=-1)) + 1e-8
        return num / den

    mask = (offset == 3).astype(x.dtype)

    if mask_2d is not None:
        mask = mask * mask_2d.astype(x.dtype)

    num = jnp.sum(x * mask)
    den = jnp.sum(mask) + 1e-8
    return num / den

def get_mid_points(pdistogram: jnp.ndarray) -> jnp.ndarray:
    """Return bin mid-points (Å)  for a given distogram tensor."""
    boundaries = jnp.linspace(2.0, 22.0, 63)                     # [2, …, 22]  共 63 端点
    lower, upper = 1.0, 27.0                                     # 1 Å, 22+5 Å
    exp_boundaries = jnp.concatenate(
        [jnp.array([lower]), boundaries, jnp.array([upper])]
    )                                                            # shape (65,)
    # 中点：(b_i + b_{i+1}) / 2.   → shape (64,)
    return (exp_boundaries[:-1] + exp_boundaries[1:]) * 0.5

@partial(jax.jit, static_argnames=('binary',))
def _get_con_loss(dgram: jnp.ndarray,
                      dgram_bins: jnp.ndarray,
                      cutoff: float | None = None,
                      *,
                      binary: bool = False) -> jnp.ndarray:
    """(L,L,64) -> (L,L) contact loss, same semantics as PyTorch version."""
    if cutoff is None:
        cutoff = dgram_bins[-1]

    within = dgram_bins < cutoff                  # (64,) bool
    big_neg = -1e7
    px      = jax.nn.softmax(dgram, axis=-1)
    px_pos  = jax.nn.softmax(jnp.where(within, dgram, dgram + big_neg), axis=-1)

    cat_ent = -(px_pos * jax.nn.log_softmax(dgram, axis=-1)).sum(axis=-1)
    bin_p   = (px * within).sum(axis=-1)
    bin_ent = -jnp.log(bin_p + 1e-8)

    w = jnp.asarray(binary, dtype=dgram.dtype)
    return w * bin_ent + (1. - w) * cat_ent       # (L,L)


def min_k(x: jnp.ndarray,
              k: int | float = None,
              mask: jnp.ndarray | None = None) -> jnp.ndarray:
    """
    取最后一维中最小的 k 个值的平均（按 mask 过滤）。
    若 k=None 或 k=inf, 则取所有可选元素。
    """
    if mask is None:
        mask = jnp.ones_like(x, dtype=bool)

    masked_x = jnp.where(mask, x, jnp.inf)
    y = jnp.sort(masked_x, axis=-1)               # 升序

    # 处理 k 的取值
    k_all = x.shape[-1]
    k_int = k_all if k is None else k

    idx     = jnp.arange(k_all)
    k_mask  = (idx < k_int)[(None,) * (y.ndim - 1) + (...,)]
    pick_m  = k_mask & jnp.isfinite(y)

    num_sel = pick_m.sum(axis=-1)
    sum_sel = jnp.where(pick_m, y, 0.).sum(axis=-1)
    return sum_sel / (num_sel + 1e-8)

@partial(jax.jit, static_argnames=('binary',))
def get_con_loss(dgram: jnp.ndarray,
                     dgram_bins: jnp.ndarray,
                     *,
                     num: int | float = None,
                     seqsep: int = 3,
                     num_pos: int | float = jnp.inf,
                     cutoff: float | None = None,
                     binary: bool = False,
                     mask_1d: jnp.ndarray | None = None,
                     mask_1b: jnp.ndarray | None = None) -> jnp.ndarray:
    """
    Shapes
    ------
    dgram        : (L, L, 64)
    mask_1d      : (L,)    bool, optional
    mask_1b      : (L, L)  bool, optional
    Return value : scalar or (L,) depending on `num_pos`
    """
    L = dgram.shape[1]

    con_loss = _get_con_loss(dgram, dgram_bins, cutoff, binary=binary)  # (L,L)

    # 序列间距 ≥ seqsep 的残基对
    idx     = jnp.arange(L)
    offset  = idx[:, None] - idx[None, :]
    m_pair  = (jnp.abs(offset) >= seqsep)

    if mask_1b is None:
        mask_1b = jnp.ones((L, L), dtype=bool)
    if mask_1d is None:
        mask_1d = jnp.ones((L,), dtype=bool)

    m_pair = m_pair & mask_1b              # (L,L) 最终 pair-mask

    # 先在列维度做 min-k：得到 (L,) 向量
    p = min_k(con_loss, k=num, mask=m_pair)   # (L,)

    # 再在残基维度做一次 min-k
    p_final = min_k(p, k=num_pos, mask=mask_1d)  # scalar 或 (L,)

    return p_final

def mask_loss(x, mask=None, mask_grad=False):
    if mask is None:
        return x.mean()
    else:
        x_masked = (x * mask).sum() / (1e-8 + mask.sum())
        if mask_grad:
            return (x.mean() - x_masked).detach() + x_masked
        else:
            return x_masked

def get_plddt_loss(lddt: jnp.ndarray, end: float = 50.0,mask_1d=None) -> jnp.ndarray:
    is_ligand = 1- mask_1d
    ligand_lddt = lddt[is_ligand,0].mean() /100

    return  1-ligand_lddt

def af3_json(pdb_sequence:str, chain_id:list, name:str, seed:list, single:bool, ligandccd:list| None, ligand_id:list,bonds = None):
    '''
    This script only protein & ligand, and no gly and modify.
    seed is a list
    single is a bool whether run data pipeline
    proteinchain: eg.["A", "B"] default is "A"
    ligandchain: eg.["C", "D"] default is "B"
    ligandccd: eg. ["RET"]
    LYR is a number means pdb file have LYR, LYR is the number of LYR
    '''
    af3_dic = {}
    af3_dic['dialect'] = "alphafold3"
    af3_dic["version"] = 1
    af3_dic['name'] = name
    af3_dic['sequences'] = []
    af3_dic['modelSeeds'] = seed
    af3_dic["bondedAtomPairs"] = bonds
    af3_dic["userCCD"]= None

    chains = str(pdb_sequence).split(',')
    for _, i in enumerate(chain_id):
        protein = {}
        protein['id'] = list(i)
        protein['sequence'] = str(chains[_])
        if single:
            protein["unpairedMsa"]=""
            protein["pairedMsa"]=""
            protein["templates"]=[]
        af3_dic['sequences'].append({"protein":protein})

    if ligandccd != None:
        ligand ={}
        ligand["ccdCodes"] = list(ligandccd)
        ligand["id"] = list(ligand_id)
        af3_dic['sequences'].append({"ligand":ligand})

    return af3_dic

def argmax2seq(argmax: np.ndarray) -> str:
    seq = []
    for i in argmax:
        for key, value in residue_names.POLYMER_TYPES_ORDER_WITH_UNKNOWN_AND_GAP.items():
            if i == value:
                if key == 'UNK':
                   continue
                else:
                  seq.append(residue_names.PROTEIN_COMMON_THREE_TO_ONE[key])
                  break
    seq = ''.join(seq)
    return seq

def update_sequence(opt,seq_logits, batch, mask, alpha=2.0, binder_chain='A'):
    batch["logits"] = alpha * seq_logits
    # X =  batch['logits']- torch.sum(torch.eye(batch['logits'].shape[-1])[[0,1,6,22,23,24,25,26,27,28,29,30,31,32]],dim=0).to(device)*(1e10)
    X = batch['logits'] - jnp.sum(jnp.eye(batch['logits'].shape[-1])[jnp.array([4, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30])], axis=0) * 1e10
    soft = jax.nn.softmax(X / opt["temp"], axis=-1)   # (..., C)
    hard_idx = jnp.argmax(soft, axis=-1)                               # (...,)
    hard_onehot = jax.nn.one_hot(hard_idx, soft.shape[-1], dtype=soft.dtype)
    hard = lax.stop_gradient(hard_onehot - soft) + soft                # straight-through

    pseudo = opt["soft"] * soft + (1.0 - opt["soft"]) * seq_logits
    pseudo = opt["hard"] * hard + (1.0 - opt["hard"]) * pseudo
    res_type = pseudo * mask + seq_logits * (1.0 - mask)

    mask         = batch["entity_id"] == 1              # (L,) bool DeviceArray
    aatype_full  = jnp.argmax(res_type, axis=-1)        # (L,) int32
    aatype   = jnp.where(mask, aatype_full, 0)   # (L,) int32
    new_batch = batch.copy()
    new_batch["soft"]     = soft
    new_batch["hard"]     = hard
    new_batch["pseudo"]   = pseudo
    new_batch["aatype"]   = aatype

    first_chain = new_batch["msa"][0,:]
    msa_first_updated = jnp.where(mask, aatype,first_chain)
    new_batch["msa"].at[0,:].set(lax.stop_gradient(msa_first_updated))
    new_batch["profile"] = lax.stop_gradient(res_type.astype(jnp.float32))
    # jax.debug.print('aatype {}', aatype)
    return new_batch,res_type



class ModelRunner:
  """Helper class to run structure prediction stages."""

  def __init__(
      self,
      config: model.Model.Config,
      device: jax.Device,
      model_dir: pathlib.Path,
  ):
    self._model_config = config
    self._device = device
    self._model_dir = model_dir
    self.o = optax.adam(1.0)
    # self.loss_fn = self._build_loss_fn()

  @functools.cached_property
  def model_params(self) -> hk.Params:
    """Loads model parameters from the model directory."""
    return params.get_model_haiku_params(model_dir=self._model_dir)

  @functools.cached_property
  def _model(
      self
  ) -> Callable[[jnp.ndarray, features.BatchDict], model.ModelResult]:
    """Loads model parameters and returns a jitted model forward pass."""

    @hk.transform
    def forward_fn(batch):
        return model.Model(self._model_config)(batch)

    return functools.partial(
        jax.jit(forward_fn.apply, device=self._device), self.model_params # haiku apply 第一个参数是params 第二个参数是rng
    )

  @functools.cached_property
  def _designmodel(
      self
  ) -> Callable[[jnp.ndarray, features.BatchDict], model.ModelResult]:
    """Loads model parameters and returns a jitted model forward pass."""

    @hk.transform
    def forward_fn(seq_logits,batch):
        # import pprint
        # pprint.pprint(self.model_params.keys())
        return model.Af3Design(self._model_config)(seq_logits, batch)
    return functools.partial(
        forward_fn.apply, self.model_params)
    return functools.partial(
        jax.jit(forward_fn.apply, device=self._device),
        self.model_params
    )


  def get_inference_feature(
      self, featurised_example: features.BatchDict, rng_key=jax.random.PRNGKey(42), af3design: bool = True
  ) -> model.ModelResult:
    """Computes a forward pass of the model on a featurised example."""
    featurised_example = jax.device_put(
        jax.tree_util.tree_map(
            jnp.asarray, utils.remove_invalidly_typed_feats(featurised_example)
        ),
        self._device,
    )
    if af3design:
        return featurised_example

    else:
        result = self._model(rng_key, featurised_example)

        result = jax.tree.map(np.asarray, result)
        result = jax.tree.map(
            lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x,
            result,
        )
        result = dict(result)
        identifier = self.model_params['__meta__']['__identifier__'].tobytes()
        result['__identifier__'] = identifier
        return result

  def diffusion_result(self, result):
        result = jax.tree.map(np.asarray, result)
        result = jax.tree.map(
            lambda x: x.astype(jnp.float16) if x.dtype == jnp.bfloat16 else x,
            result,
        )
        result = dict(result)
        identifier = self.model_params['__meta__']['__identifier__'].tobytes()
        result['__identifier__'] = identifier
        return result


  def extract_inference_results_and_maybe_embeddings(
      self,
      batch: features.BatchDict,
      result: model.ModelResult,
      target_name: str,
  ) -> tuple[list[model.InferenceResult], dict[str, np.ndarray] | None]:
    """Extracts inference results and embeddings (if set) from model outputs."""
    inference_results = list(
        model.Model.get_inference_result(
            batch=batch, result=result, target_name=target_name
        )
    )
    num_tokens = len(inference_results[0].metadata['token_chain_ids'])
    embeddings = {}
    if 'single_embeddings' in result:
      embeddings['single_embeddings'] = result['single_embeddings'][:num_tokens]
    if 'pair_embeddings' in result:
      embeddings['pair_embeddings'] = result['pair_embeddings'][
          :num_tokens, :num_tokens
      ]
    return inference_results, embeddings or None

  def updata_seq(self, seq_logits, opt):
      step = opt['step']
      iteration = opt['iteration']
      t = opt['t']
      stage = opt['stage']

      def stage_1_fn(seq_logits):
          lambda_ = (step + 1) / iteration
          return (1 - lambda_) * seq_logits + lambda_ * jax.nn.softmax(seq_logits / t)

      def stage_2_fn(seq_logits):
          temperature_initial = 1e-2
          temperature = temperature_initial + (1 - temperature_initial) * (1 - (step + 1) / iteration)**2
          return jax.nn.softmax(seq_logits / temperature)

      def stage_3_fn(seq_logits):
          softmax_logits = jax.nn.softmax(seq_logits)
          final_sequence = jax.nn.one_hot(jnp.argmax(softmax_logits), softmax_logits.shape[-1]) - softmax_logits
          return jax.lax.stop_gradient(final_sequence) + softmax_logits

      seq_logits = lax.cond(
          stage == 1,
          stage_1_fn,
          lambda _: lax.cond(
              stage == 2,
              stage_2_fn,
              lambda _: stage_3_fn(seq_logits),
              seq_logits
          ),
          seq_logits
      )

      return seq_logits


  def get_model(self,pre_run):
      @remat # jit 省去速度差别非常大
      def _model(seq_logits, batch, rng):

        batch,res_type = update_sequence(batch['update_sequence']['opt'], seq_logits,batch, batch['update_sequence']['mask'])
        result=  self._designmodel(rng, res_type, batch)
        chain_mask = batch['chain_mask'] # protein is 1, ligand is 0
        mask_2d = chain_mask[ :, None] * chain_mask[ None, :]
        plddt_loss = get_plddt_loss(result['predicted_lddt'],mask_1d=chain_mask) #, batch['is_ligand'], example)
        pdist = result['distogram']['pdist']
        mid_pts = get_mid_points(pdist)                       # (64,)
        num_optimizing_binder_pos = batch['update_sequence']['opt']["num_optimizing_binder_pos"]
        # pdist_protein = pdist * mask_2d[:, :, None]
        # pdist_inter =
        con_loss = get_con_loss(pdist, mid_pts,
                            num=4, seqsep=9, cutoff=14,
                            binary=False,
                            mask_1d=chain_mask, mask_1b=chain_mask)
        i_con_loss = get_con_loss(pdist, mid_pts,
                                        num=2, seqsep=0, #num_pos=num_optimizing_binder_pos,
                                        cutoff=21, binary=False,
                                        mask_1d=chain_mask, mask_1b=1-chain_mask)
        helix_loss = _get_helix_loss(pdist, mid_pts,
                                    offset=None, mask_2d=mask_2d, binary=True)
        losses = {
            'con_loss': con_loss,
            'helix_loss': helix_loss,
        }
        if not pre_run:
           losses.update({
                'plddt_loss': plddt_loss,
                'i_con_loss': i_con_loss,
            })
        loss_scales = {
            'con_loss': 0.5,
            'i_con_loss': 1.0,
            'helix_loss': 0.5,
            'plddt_loss': 0.0,
            'pae_loss': 0.0,
            'i_pae_loss': 0.0,
            'rg_loss': 0.0,
        }
        total_loss = sum(loss * loss_scales[name] for name, loss in losses.items())
        aux = {}
        aux['losses'] = losses
        aux['batch'] = batch
        aux['result'] = result
        # total_loss = jnp.mean(jnp.abs(pdist)) # tg
        return total_loss,aux
      return jax.value_and_grad(_model, argnums=0,has_aux=True)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ResultsForSeed:
  """Stores the inference results (diffusion samples) for a single seed.

  Attributes:
    seed: The seed used to generate the samples.
    inference_results: The inference results, one per sample.
    full_fold_input: The fold input that must also include the results of
      running the data pipeline - MSA and templates.
    embeddings: The final trunk single and pair embeddings, if requested.
  """

  seed: int
  inference_results: Sequence[model.InferenceResult]
  full_fold_input: folding_input.Input
  embeddings: dict[str, np.ndarray] | None = None

def predict_structure(
    fold_input: folding_input.Input,
    model_runner: ModelRunner,
    buckets: Sequence[int] | None = None,
    ref_max_modified_date: datetime.date | None = None,
    conformer_max_iterations: int | None = None,
) -> Sequence[ResultsForSeed]:
  """Runs the full inference pipeline to predict structures for each seed."""

  print(f'Featurising data with {len(fold_input.rng_seeds)} seed(s)...')
  featurisation_start_time = time.time()
  ccd = chemical_components.cached_ccd(user_ccd=fold_input.user_ccd)
  featurised_examples = featurisation.featurise_input(
      fold_input=fold_input,
      buckets=buckets,
      ccd=ccd,
      verbose=True,
      ref_max_modified_date=ref_max_modified_date,
      conformer_max_iterations=conformer_max_iterations,
  )
  print(
      f'Featurising data with {len(fold_input.rng_seeds)} seed(s) took'
      f' {time.time() - featurisation_start_time:.2f} seconds.'
  )
  print(
      'Running model inference and extracting output structure samples with'
      f' {len(fold_input.rng_seeds)} seed(s)...'
  )
  all_inference_start_time = time.time()
  all_inference_results = []
  for seed, example in zip(fold_input.rng_seeds, featurised_examples):
    print(f'Running model inference with seed {seed}...')
    inference_start_time = time.time()
    rng_key = jax.random.PRNGKey(seed)
    result = model_runner.get_inference_feature(example, rng_key, af3design=False)

    print(
        f'Running model inference with seed {seed} took'
        f' {time.time() - inference_start_time:.2f} seconds.'
    )
    print(f'Extracting inference results with seed {seed}...')
    extract_structures = time.time()
    inference_results, embeddings = (
        model_runner.extract_inference_results_and_maybe_embeddings(
            batch=example, result=result, target_name=fold_input.name
        )
    )
    print(
        f'Extracting {len(inference_results)} inference samples with'
        f' seed {seed} took {time.time() - extract_structures:.2f} seconds.'
    )

    all_inference_results.append(
        ResultsForSeed(
            seed=seed,
            inference_results=inference_results,
            full_fold_input=fold_input,
            embeddings=embeddings,
        )
    )
  print(
      'Running model inference and extracting output structures with'
      f' {len(fold_input.rng_seeds)} seed(s) took'
      f' {time.time() - all_inference_start_time:.2f} seconds.'
  )
  return all_inference_results



def get_example(
    fold_input: folding_input.Input,
    buckets: Sequence[int] | None = None,
    ref_max_modified_date: datetime.date | None = None,
    conformer_max_iterations: int | None = None,
) -> Sequence[ResultsForSeed]:
  """Runs the full inference pipeline to predict structures for each seed."""

  ccd = chemical_components.cached_ccd(user_ccd=fold_input.user_ccd)
  featurised_examples = featurisation.featurise_input(
      fold_input=fold_input,
      buckets=buckets,
      ccd=ccd,
      verbose=True,
      ref_max_modified_date=ref_max_modified_date,
      conformer_max_iterations=conformer_max_iterations,
  )
  return featurised_examples[0]


chain_to_number = {
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
    'E': 5,
    'F': 6,
    'G': 7,
    'H': 8,
    'I': 9,
    'J': 10,
}

def norm_seq_grad(grad, chain_mask, eps=1e-7):
    #  Two-dimensional case: grad (L, C)
    if grad.ndim == 2:
        mask = chain_mask.astype(bool)          # (L,)
        masked = grad * mask[:, None]                           # (L,C)
        eff_L = jnp.sum((jnp.sum(masked**2, axis=-1) > 0))      # scalar
        gn = jnp.linalg.norm(masked)                            # scalar
        scale = jnp.sqrt(eff_L) / (gn + eps)
        return grad * scale                                     # (L,C)

    #  Three-dimensional case: grad (B, L, C)
    mask = chain_mask.astype(bool).reshape((1, -1, 1))          # (1,L,1)
    masked = grad * mask                                        # (B,L,C)
    eff_L = jnp.sum((jnp.sum(masked**2, axis=-1, keepdims=True) > 0),
                    axis=-2, keepdims=True)                     # (B,1,1)
    gn = jnp.linalg.norm(masked, axis=(-1, -2), keepdims=True)  # (B,1,1)
    return grad * jnp.sqrt(eff_L.astype(grad.dtype)) / (gn + eps)

def af3_hallucination(
    model_runner: ModelRunner,
    pre_run:False,
    input_res_type=None,
    stage='3stage',
    pre_run_iters=25,
    soft_iteration=100,
    temp_iteration=40,
    hard_iteration=5,
    e_soft=1.0,
    alpha=2.0,
    binder_chain='A',
    noise_scaling=0.1,
    optimizer_type='sgd',
    learning_rate=0.1,
    learning_rate_pre=0.1,
    losses=None,
):
    length = opt_af3deisgn.binder_length
    ligand_ccd = opt_af3deisgn.target
    motif_pos = opt_af3deisgn.motif_pos
    alphabet = 'ARNDCQEGHILKMFPSTWYVXXXXXXXXXXXXX'

    if motif_pos is not None:
        seq = 'X' * (motif_pos-1) + 'K' + 'X' * (length - motif_pos)
    else:
        seq = 'X' * length

    fold_input_dic = af3_json(pdb_sequence=seq, chain_id=['A'], name=opt_af3deisgn.name, seed=[42], single=True, ligandccd=ligand_ccd, ligand_id=['B'],bonds=opt_af3deisgn.af3_bonds)
    fold_inputs = folding_input.load_fold_inputs_from_path(fold_input_dic )
    for fold_input in fold_inputs:
        # if _NUM_SEEDS.value is not None:
        #     print(f'Expanding fold job {fold_input.name} to {_NUM_SEEDS.value} seeds')
        #     fold_input = fold_input.with_multiple_seeds(_NUM_SEEDS.value)
        example = get_example(
            fold_input=fold_input,
            buckets=tuple(int(bucket)for bucket in opt_af3deisgn._BUCKETS),#for bucket in _BUCKETS.value),
            conformer_max_iterations=None, #opt._CONFORMER_MAX_ITERATIONS.value,
        ) # get feature information
        batch = model_runner.get_inference_feature(example)

    seq_logits = jnp.zeros((batch['aatype'].shape[0], 31), dtype=jnp.float32)
    # seq_logits,batch = model_runner.get_inference_feature(example)
    chain_mask = (batch["entity_id"] == chain_to_number[binder_chain]).astype(jnp.int32)
    if pre_run:
        g_shape = (chain_mask.shape[0], seq_logits.shape[-1])       # (N, C)
        gumbel = jr.gumbel(jax.random.PRNGKey(42), g_shape)

        ban_mask = jnp.take(jnp.eye(seq_logits.shape[-1], dtype=jnp.float32), jnp.array([4, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]), axis=0).sum(0)
        ban_penalty = ban_mask * 1e10
        logits_noise = noise_scaling * jax.nn.softmax(gumbel - ban_penalty, axis=-1)
        seq_logits  = jnp.where(chain_mask[:,None],logits_noise, seq_logits)
        if motif_pos is not None:
            motif_pos_aa = alphabet.index(str(opt_af3deisgn.motif_pos_aa).upper())
            seq_logits = seq_logits.at[motif_pos-1, :].set(0.0)
            seq_logits = seq_logits.at[motif_pos-1, [motif_pos_aa]].set(1.0)  # K
    else:
        seq_logits  = input_res_type

    if optimizer_type == 'sgd':
        optimizer = optax.inject_hyperparams(optax.sgd)(
            learning_rate=1.0,
            momentum=None
        )
    else:
        raise ValueError(f'Unknown optimizer type: {optimizer_type}')

    opt_state = optimizer.init(seq_logits )


    mask = jnp.ones_like(seq_logits )
    rows_to_zero = batch["entity_id"] != chain_to_number[binder_chain]
    mask = mask.at[rows_to_zero, :].set(0)
    # chain_mask = (batch["entity_id"] == chain_to_number[binder_chain]).astype(jnp.int32) # binder chain 1
    batch['chain_mask'] = chain_mask
    grad_fn = model_runner.get_model(pre_run)
    # state = model_runner.o.init(seq_logits)
    if losses is None:
        losses = {'con_loss': []}


    def design(
        seq_logits,
        batch,
        mask,
        opt_state,
        stage_name=None,
        iters = None,
        soft=0.0, e_soft=None,
        step=1.0, e_step=None,
        temp=1.0, e_temp=None,
        hard=0.0, e_hard=None,
        num_optimizing_binder_pos=1, e_num_optimizing_binder_pos=1,
        learning_rate=1.0,
        losses = None,
    ):
        def write_cif(example,
                        name: str,
                        result,
                        output_dir):
            inference_results, embeddings = (
                model_runner.extract_inference_results_and_maybe_embeddings(
                    batch=example, result=result, target_name=name
                )
            )
            r_results = ResultsForSeed(
                        seed=jax.random.PRNGKey(42),
                        full_fold_input=fold_input,
                        inference_results=inference_results,
                        embeddings=embeddings,
                    )
            # r_results.inference_results is list
            for rr in r_results.inference_results:
                post_processing.write_output(
                    inference_result=rr,
                    output_dir=output_dir,
                    name=name,
                )
        m = {"soft":[soft,e_soft],"temp":[temp,e_temp],"hard":[hard,e_hard], "step":[step,e_step], 'num_optimizing_binder_pos':[num_optimizing_binder_pos, e_num_optimizing_binder_pos]}
        m = {k:[s,(s if e is None else e)] for k,(s,e) in m.items()}

        opt = {}
        for i in range(iters):
            for k,(s,e) in m.items():
                if k == "temp":
                    opt[k] = (e+(s-e)*(1-(i)/iters)**2)
                else:
                    v = (s+(e-s)*((i)/iters))
                    if k == "step": step = v
                    opt[k] = v

            lr_scale = step * ((1 - opt["soft"]) + (opt["soft"] * opt["temp"]))
            opt["num_optimizing_binder_pos"] = jnp.array(opt["num_optimizing_binder_pos"], int)

            lr = opt["lr_rate"] = learning_rate * lr_scale
            # batch = update_sequence(opt, seq_logits,batch, mask)
            batch['update_sequence'] ={}
            batch['update_sequence']['opt'] = opt
            batch['update_sequence']['mask'] = mask
            key, rng = jax.random.split(jax.random.PRNGKey(42), 2)
            (loss, aux), grad= grad_fn(seq_logits, batch, rng)
            batch = aux['batch']

            # name = f'{stage_name}_{i}'
            # out_dir = '/home/ge/app/af3design/af3design_test_logits_onlycon'
            # if os.path.exists(out_dir) is False:
            #     os.makedirs(out_dir, exist_ok=True)
            aux['result'] = model_runner.diffusion_result(aux['result'])
            # write_cif(example=example,name=name,result=aux['result'],output_dir=out_dir)


            print(aux['losses'])
            # losses['con_loss'].append(aux['losses']['con_loss'])

            grad = grad.at[batch['entity_id']!=chain_to_number[binder_chain],:].set(0.0)
            if motif_pos is not None:
                grad = grad.at[[motif_pos-1],:].set(0.0)
            grad = grad.at[:, [4, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]].set(0.0)
            grad = norm_seq_grad(grad, batch['chain_mask'])

            if jnp.isnan(aux['result']['distogram']['probs_logits']).any():
                print('NaN in distogram logits, skipping update')
                quit()

            @partial(jax.jit)
            def update_grad(state, grad, params):
                updates, state = optimizer.update(grad, state, params)
                grad = jax.tree.map(lambda x:-x, updates)
                return grad,state
            updates, opt_state = update_grad( opt_state,grad, seq_logits)#,learning_rate=jnp.asarray(lr, jnp.float32))

            seq_logits = jax.tree_util.tree_map(lambda x,g:x-lr*g, seq_logits, updates)

            print('-'   *100)
            print(f"Epoch {i}: lr: {lr:.3f}, soft: {opt['soft']:.2f}, hard: {opt['hard']:.2f}, temp: {opt['temp']:.2f}, total loss: {loss.item():.2f}")

        best_arg = batch["aatype"][chain_mask.astype(bool)]
        best_seq = ''.join(alphabet[i] for i in best_arg)
        best_seq = argmax2seq(best_arg)
        print(best_seq)
        return seq_logits, batch, opt_state, losses,best_seq

    if pre_run:
       seq_logits,batch,opt_state, losses,best_seq = design(seq_logits, batch=batch, mask=mask,opt_state=opt_state,stage_name='pre_run',iters=pre_run_iters, soft=1.0, learning_rate=0.2,losses=losses)
       print('a',seq_logits.shape)
       return seq_logits, batch, losses
    elif stage == '3stage':
        print('-'*100)
        print(f"logits to softmax(T={e_soft})")
        print('-'*100)
        seq_logits,batch,opt_state,losses,best_seq = design(seq_logits, batch=batch, mask=mask,opt_state=opt_state,stage_name='soft',learning_rate=learning_rate,iters=soft_iteration, e_soft=e_soft,num_optimizing_binder_pos=1, e_num_optimizing_binder_pos=8,losses=losses)
        print('-'*100)
        print("softmax(T=1) to softmax(T=0.01)")
        print('-'*100)
        print("set res_type_logits to logits")
        optimizer = optax.inject_hyperparams(optax.sgd)(
            learning_rate=1.0,
            momentum=None
        )
        new_logits = seq_logits * alpha
        opt_state = optimizer.init(new_logits)

        seq_logits,batch,opt_state,losses,best_seq= design(new_logits, batch=batch, mask=mask, opt_state=opt_state,stage_name='soft_t',learning_rate=learning_rate, iters=temp_iteration, soft=1.0, temp = 1.0,e_temp=0.01,num_optimizing_binder_pos=8, e_num_optimizing_binder_pos=12,losses=losses)
        print('-'*100)
        print("hard")
        print('-'*100)
        seq_logits,batch,opt_state,losses,best_seq = design(seq_logits, batch=batch, mask=mask, opt_state=opt_state,stage_name='hard',learning_rate=learning_rate, iters=hard_iteration, soft=1.0, hard = 1.0,temp=0.01, num_optimizing_binder_pos=12, e_num_optimizing_binder_pos=16,losses=losses)
        return losses,best_seq
    elif stage == 'logits':
        seq_logits,batch,opt_state,losses,best_seq = design(seq_logits, batch=batch, mask=mask,opt_state=opt_state,stage_name='logits',learning_rate=learning_rate,iters=soft_iteration, soft = 0.0, e_soft=0.0,losses=losses)
        return losses,best_seq

def design(model_runner: ModelRunner,example):
    def write_cif(example,
                    name: str,
                    aux,
                    output_dir):
        inference_results, embeddings = (
            model_runner.extract_inference_results_and_maybe_embeddings(
                batch=example, result=aux, target_name=name
            )
        )
        r_results = ResultsForSeed(
                    inference_results=inference_results,
                    embeddings=embeddings,
                )
        # r_results.inference_results is list
        for rr in r_results.inference_results:
            post_processing.write_output(
                inference_result=rr,
                output_dir=output_dir,
                name=name,
            )
    # seed = fold_input.rng_seeds[0]
    seq_logits,batch = model_runner.get_inference_feature(example) # logits shape [L,31]
    stage_dict = {1: 50, 2: 15, 3: 1}
    optimizer = jax.jit(model_runner.update_grad)
    for iter in range(10):
        print(f'iter: {iter}')
        state = model_runner.o.init(seq_logits)
        for stage in range(1,4):
            iteration = stage_dict[stage]
            schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.01,
            peak_value=1,
            warmup_steps=iteration/2,
            decay_steps=iteration)
            for step in  range(iteration):
                batch['design_opt'] = {}
                batch['design_opt']['step'] = step
                batch['design_opt']['iteration'] = iteration
                batch['design_opt']['t'] = 1
                batch['design_opt']['stage'] = stage

                grad_fn = model_runner.get_model(example)

                (loss, aux), grad= grad_fn(seq_logits, batch,jax.random.PRNGKey(42))
                aux = model_runner.diffusion_result(aux)
                state,grad = optimizer(grad, seq_logits, state)
                lr =  schedule(step)

                seq_logits = jax.tree_util.tree_map(lambda x,g:x-lr*g, seq_logits, grad)

                print(f'step: {step}, iter: {iter}, loss: {loss}')
                print(argmax2seq(jnp.argmax(seq_logits[:,:20], axis=-1)))
        name = f'iter{iter}'
        out_dir = '/home/ge/app/af3design/test'
        write_cif(example=example,name=name,aux=aux,output_dir=out_dir)
        input_cif = f'{out_dir}/{name}_model.cif'
        out_pdb = f'{out_dir}/{name}_model.pdb'
        cif2pdb(input_cif, out_pdb)
        result = subprocess.run(['/data/ge/conda/envs/mpnn_env/bin/python','/home/ge/app/LigandMPNN/score.py',
                        f'--pdb_path',out_pdb,
                        f'--model_type','ligand_mpnn',
                        f'--out_folder', out_dir], check=True, capture_output=True)
        mpnn_logits = np.load(f'{out_dir}/{name}_model.npy') #logtis shape [B,L,21]

        seq_logits = seq_logits.at[:mpnn_logits.shape[1], :mpnn_logits.shape[2]].set(jnp.array(mpnn_logits[0]))
        print(f'step: {step}, iter: {iter}, loss: {loss}')
        print(argmax2seq(jnp.argmax(seq_logits[:,:21], axis=-1)))

def write_fold_input_json(
    fold_input: folding_input.Input,
    output_dir: os.PathLike[str] | str,
) -> None:
  """Writes the input JSON to the output directory."""
  os.makedirs(output_dir, exist_ok=True)
  path = os.path.join(output_dir, f'{fold_input.sanitised_name()}_data.json')
  print(f'Writing model input JSON to {path}')
  with open(path, 'wt') as f:
    f.write(fold_input.to_json())

def write_outputs(
    all_inference_results: Sequence[ResultsForSeed],
    output_dir: os.PathLike[str] | str,
    job_name: str,
) -> None:
  """Writes outputs to the specified output directory."""
  ranking_scores = []
  max_ranking_score = None
  max_ranking_result = None

  output_terms = (
      pathlib.Path(alphafold3.cpp.__file__).parent / 'OUTPUT_TERMS_OF_USE.md'
  ).read_text()

  os.makedirs(output_dir, exist_ok=True)
  for results_for_seed in all_inference_results:
    seed = results_for_seed.seed
    for sample_idx, result in enumerate(results_for_seed.inference_results):
      sample_dir = os.path.join(output_dir, f'seed-{seed}_sample-{sample_idx}')
      os.makedirs(sample_dir, exist_ok=True)
      post_processing.write_output(
          inference_result=result,
          output_dir=sample_dir,
          name=f'{job_name}_seed-{seed}_sample-{sample_idx}',
      )
      ranking_score = float(result.metadata['ranking_score'])
      ranking_scores.append((seed, sample_idx, ranking_score))
      if max_ranking_score is None or ranking_score > max_ranking_score:
        max_ranking_score = ranking_score
        max_ranking_result = result

    if embeddings := results_for_seed.embeddings:
      embeddings_dir = os.path.join(output_dir, f'seed-{seed}_embeddings')
      os.makedirs(embeddings_dir, exist_ok=True)
      post_processing.write_embeddings(
          embeddings=embeddings,
          output_dir=embeddings_dir,
          name=f'{job_name}_seed-{seed}',
      )

  if max_ranking_result is not None:  # True iff ranking_scores non-empty.
    post_processing.write_output(
        inference_result=max_ranking_result,
        output_dir=output_dir,
        # The output terms of use are the same for all seeds/samples.
        terms_of_use=output_terms,
        name=job_name,
    )
    # Save csv of ranking scores with seeds and sample indices, to allow easier
    # comparison of ranking scores across different runs.
    with open(
        os.path.join(output_dir, f'{job_name}_ranking_scores.csv'), 'wt'
    ) as f:
      writer = csv.writer(f)
      writer.writerow(['seed', 'sample', 'ranking_score'])
      writer.writerows(ranking_scores)

def replace_db_dir(path_with_db_dir: str, db_dirs: Sequence[str]) -> str:
  """Replaces the DB_DIR placeholder in a path with the given DB_DIR."""
  template = string.Template(path_with_db_dir)
  if 'DB_DIR' in template.get_identifiers():
    for db_dir in db_dirs:
      path = template.substitute(DB_DIR=db_dir)
      if os.path.exists(path):
        return path
    raise FileNotFoundError(
        f'{path_with_db_dir} with ${{DB_DIR}} not found in any of {db_dirs}.'
    )
  if not os.path.exists(path_with_db_dir):
    raise FileNotFoundError(f'{path_with_db_dir} does not exist.')
  return path_with_db_dir

def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: ModelRunner | None,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
    ref_max_modified_date: datetime.date | None = None,
    conformer_max_iterations: int | None = None,
    force_output_dir: bool = False,
) -> folding_input.Input | Sequence[ResultsForSeed]:
  """Runs data pipeline and/or inference on a single fold input.

  Args:
    fold_input: Fold input to process.
    data_pipeline_config: Data pipeline config to use. If None, skip the data
      pipeline.
    model_runner: Model runner to use. If None, skip inference.
    output_dir: Output directory to write to.
    buckets: Bucket sizes to pad the data to, to avoid excessive re-compilation
      of the model. If None, calculate the appropriate bucket size from the
      number of tokens. If not None, must be a sequence of at least one integer,
      in strictly increasing order. Will raise an error if the number of tokens
      is more than the largest bucket size.
    ref_max_modified_date: Optional maximum date that controls whether to allow
      use of model coordinates for a chemical component from the CCD if RDKit
      conformer generation fails and the component does not have ideal
      coordinates set. Only for components that have been released before this
      date the model coordinates can be used as a fallback.
    conformer_max_iterations: Optional override for maximum number of iterations
      to run for RDKit conformer search.
    force_output_dir: If True, do not create a new output directory even if the
      existing one is non-empty. Instead use the existing output directory and
      potentially overwrite existing files. If False, create a new timestamped
      output directory instead if the existing one is non-empty.

  Returns:
    The processed fold input, or the inference results for each seed.

  Raises:
    ValueError: If the fold input has no chains.
  """
  print(f'\nRunning fold job {fold_input.name}...')

  if not fold_input.chains:
    raise ValueError('Fold input has no chains.')

  if (
      not force_output_dir
      and os.path.exists(output_dir)
      and os.listdir(output_dir)
  ):
    new_output_dir = (
        f'{output_dir}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    print(
        f'Output will be written in {new_output_dir} since {output_dir} is'
        ' non-empty.'
    )
    output_dir = new_output_dir
  else:
    print(f'Output will be written in {output_dir}')

  if data_pipeline_config is None:
    print('Skipping data pipeline...')
  else:
    print('Running data pipeline...')
    fold_input = pipeline.DataPipeline(data_pipeline_config).process(fold_input)

  write_fold_input_json(fold_input, output_dir)
  if model_runner is None:
    print('Skipping model inference...')
    output = fold_input
  else:
    print(
        f'Predicting 3D structure for {fold_input.name} with'
        f' {len(fold_input.rng_seeds)} seed(s)...'
    )
    all_inference_results = predict_structure(
        fold_input=fold_input,
        model_runner=model_runner,
        buckets=buckets,
        ref_max_modified_date=ref_max_modified_date,
        conformer_max_iterations=conformer_max_iterations,
    ) #正常的token长度

    print(f'Writing outputs with {len(fold_input.rng_seeds)} seed(s)...')
    write_outputs(
        all_inference_results=all_inference_results,
        output_dir=output_dir,
        job_name=fold_input.sanitised_name(),
    )
    output = all_inference_results

  print(f'Fold job {fold_input.name} done, output written to {output_dir}\n')
  return output


def main(_):


  # Make sure we can create the output directory before running anything.
  try:
    os.makedirs(opt_af3deisgn._OUTPUT_DIR, exist_ok=True)
  except OSError as e:
    print(f'Failed to create output directory {opt_af3deisgn._OUTPUT_DIR}: {e}')
    raise

  if True:
    # Fail early on incompatible devices, but only if we're running inference.
    gpu_devices = jax.local_devices(backend='gpu')
    if gpu_devices:
      compute_capability = float(
          gpu_devices[opt_af3deisgn._GPU_DEVICE].compute_capability
      )
      if compute_capability < 6.0:
        raise ValueError(
            'AlphaFold 3 requires at least GPU compute capability 6.0 (see'
            ' https://developer.nvidia.com/cuda-gpus).'
        )
      elif 7.0 <= compute_capability < 8.0:
        xla_flags = os.environ.get('XLA_FLAGS')
        required_flag = '--xla_disable_hlo_passes=custom-kernel-fusion-rewriter'
        if not xla_flags or required_flag not in xla_flags:
          raise ValueError(
              'For devices with GPU compute capability 7.x (see'
              ' https://developer.nvidia.com/cuda-gpus) the ENV XLA_FLAGS must'
              f' include "{required_flag}".'
          )
        # if _FLASH_ATTENTION_IMPLEMENTATION.value != 'xla':
        #   raise ValueError(
        #       'For devices with GPU compute capability 7.x (see'
        #       ' https://developer.nvidia.com/cuda-gpus) the'
        #       ' --flash_attention_implementation must be set to "xla".'
        #   )

  data_pipeline_config = None

  if True:
    devices = jax.local_devices(backend='gpu')
    print(
        f'Found local devices: {devices}, using device {opt_af3deisgn._GPU_DEVICE}:'
        f' {devices[opt_af3deisgn._GPU_DEVICE]}'
    )

    print('Building model from scratch...')
    model_runner = ModelRunner(
        config=make_model_config(
            flash_attention_implementation=typing.cast(
                attention.Implementation, opt_af3deisgn._FLASH_ATTENTION_IMPLEMENTATION
            ),
            num_diffusion_samples=opt_af3deisgn._NUM_DIFFUSION_SAMPLES,
            num_recycles=opt_af3deisgn._NUM_RECYCLES,
            return_embeddings=opt_af3deisgn._SAVE_EMBEDDINGS,
        ),
        device=devices[opt_af3deisgn._GPU_DEVICE],
        model_dir=pathlib.Path(opt_af3deisgn.MODEL_DIR),
    )
    # Check we can load the model parameters before launching anything.
    print('Checking that model parameters can be loaded...')
    _ = model_runner.model_params


  print('warm up')
  seq_logits, batch, losses = af3_hallucination(model_runner,pre_run=True,input_res_type = None,stage=None,losses=None,)
  print('warm up done')
  losses,seq = af3_hallucination(model_runner,pre_run=False,input_res_type = seq_logits,stage=opt_af3deisgn.stage,losses=losses,)

  fold_input_dic = af3_json(pdb_sequence=seq, chain_id=['A'], name=opt_af3deisgn.name, seed=[42], single=True, ligandccd=opt_af3deisgn.target, ligand_id=['B'],bonds=opt_af3deisgn.af3_bonds)
  fold_inputs = folding_input.load_fold_inputs_from_path(fold_input_dic )
  for fold_input in fold_inputs:
        process_fold_input(
        fold_input=fold_input,
        data_pipeline_config=data_pipeline_config,
        model_runner=model_runner,
        output_dir=os.path.join(opt_af3deisgn._OUTPUT_DIR, fold_input.sanitised_name()),
        buckets=tuple(int(bucket) for bucket in opt_af3deisgn._BUCKETS),
        conformer_max_iterations=None,
        force_output_dir=False
        )




if __name__ == '__main__':
  # flags.mark_flags_as_required(['output_dir'])
  app.run(main)

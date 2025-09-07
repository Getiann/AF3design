# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""AlphaFold 3 structure prediction script.

AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

To request access to the AlphaFold 3 model parameters, follow the process set
out at https://github.com/google-deepmind/alphafold3. You may only use these
if received directly from Google. Use is subject to terms of use available at
https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
"""

from collections.abc import Callable, Sequence
import csv
import dataclasses
import datetime
import functools
import multiprocessing
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_CLIENT_MEM_FRACTION"] = "0.95"
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
from alphafold3.constants import residue_names
import alphafold3.cpp
from alphafold3.data import featurisation
from alphafold3.data import pipeline
from alphafold3.jax.attention import attention
from alphafold3.model import feat_batch
from alphafold3.constants import residue_names
from alphafold3.model import features
from alphafold3.model import model
from alphafold3.model import params
from alphafold3.model import post_processing
from alphafold3.model.components import utils
import sys
# sys.path.append('/home/ge/app/LigandMPNN')
# from run_mpnn import main as run_mpnn
import subprocess
from alphafold3.model.atom_layout import atom_layout
import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np
import optax
from jax import remat
from jax import lax
_HOME_DIR = pathlib.Path(os.environ.get('HOME'))
_DEFAULT_MODEL_DIR = _HOME_DIR / 'models'
_DEFAULT_DB_DIR = _HOME_DIR / 'public_databases'


_JSON_PATH = flags.DEFINE_string(
    'json_path',
    None,
    'Path to the input JSON file.',
)
_INPUT_DIR = flags.DEFINE_string(
    'input_dir',
    None,
    'Path to the directory containing input JSON files.',
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    None,
    'Path to a directory where the results will be saved.',
)
MODEL_DIR = flags.DEFINE_string(
    'model_dir',
    _DEFAULT_MODEL_DIR.as_posix(),
    'Path to the model to use for inference.',
)

# Control which stages to run.
_RUN_DATA_PIPELINE = flags.DEFINE_bool(
    'run_data_pipeline',
    True,
    'Whether to run the data pipeline on the fold inputs.',
)
_RUN_INFERENCE = flags.DEFINE_bool(
    'run_inference',
    True,
    'Whether to run inference on the fold inputs.',
)

# Database paths.
DB_DIR = flags.DEFINE_multi_string(
    'db_dir',
    (_DEFAULT_DB_DIR.as_posix(),),
    'Path to the directory containing the databases. Can be specified multiple'
    ' times to search multiple directories in order.',
)

_CONFORMER_MAX_ITERATIONS = flags.DEFINE_integer(
    'conformer_max_iterations',
    None,  # Default to RDKit default parameters value.
    'Optional override for maximum number of iterations to run for RDKit '
    'conformer search.',
)

# JAX inference performance tuning.
_JAX_COMPILATION_CACHE_DIR = flags.DEFINE_string(
    'jax_compilation_cache_dir',
    None,
    'Path to a directory for the JAX compilation cache.',
)
_GPU_DEVICE = flags.DEFINE_integer(
    'gpu_device',
    0,
    'Optional override for the GPU device to use for inference. Defaults to the'
    ' 1st GPU on the system. Useful on multi-GPU systems to pin each run to a'
    ' specific GPU.',
)
_BUCKETS = flags.DEFINE_list(
    'buckets',
    # pyformat: disable
    ['64','120','128','256', '512', '768', '1024', '1280', '1536', '2048', '2560', '3072',
     '3584', '4096', '4608', '5120'],
    # pyformat: enable
    'Strictly increasing order of token sizes for which to cache compilations.'
    ' For any input with more tokens than the largest bucket size, a new bucket'
    ' is created for exactly that number of tokens.',
)
_FLASH_ATTENTION_IMPLEMENTATION = flags.DEFINE_enum(
    'flash_attention_implementation',
    default='xla',
    enum_values=['triton', 'cudnn', 'xla'],
    help=(
        "Flash attention implementation to use. 'triton' and 'cudnn' uses a"
        ' Triton and cuDNN flash attention implementation, respectively. The'
        ' Triton kernel is fastest and has been tested more thoroughly. The'
        " Triton and cuDNN kernels require Ampere GPUs or later. 'xla' uses an"
        ' XLA attention implementation (no flash attention) and is portable'
        ' across GPU devices.'
    ),
)
_NUM_RECYCLES = flags.DEFINE_integer(
    'num_recycles',
    1,
    'Number of recycles to use during inference.',
    lower_bound=1,
)
_NUM_DIFFUSION_SAMPLES = flags.DEFINE_integer(
    'num_diffusion_samples',
    1,
    'Number of diffusion samples to generate.',
    lower_bound=1,
)
_NUM_SEEDS = flags.DEFINE_integer(
    'num_seeds',
    None,
    'Number of seeds to use for inference. If set, only a single seed must be'
    ' provided in the input JSON. AlphaFold 3 will then generate random seeds'
    ' in sequence, starting from the single seed specified in the input JSON.'
    ' The full input JSON produced by AlphaFold 3 will include the generated'
    ' random seeds. If not set, AlphaFold 3 will use the seeds as provided in'
    ' the input JSON.',
    lower_bound=1,
)

# Output controls.
_SAVE_EMBEDDINGS = flags.DEFINE_bool(
    'save_embeddings',
    True,
    'Whether to save the final trunk single and pair embeddings in the output.',
)
_FORCE_OUTPUT_DIR = flags.DEFINE_bool(
    'force_output_dir',
    False,
    'Whether to force the output directory to be used even if it already exists'
    ' and is non-empty. Useful to set this to True to run the data pipeline and'
    ' the inference separately, but use the same output directory.',
)


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

import jax
import jax.numpy as jnp
import haiku as hk
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

def contact_loss_dgram_old(
    dgram_logits: jnp.ndarray,        # [N, N, num_bins]
    bin_edges: jnp.ndarray,           # [num_bins - 1]
    residue_chain: jnp.ndarray,       # [N], 每个 residue 属于哪个链 (int)
    intra_cutoff: float = 14.0,
    inter_cutoff: float = 22.0,
) -> jnp.ndarray:
    """
    Entropy-based contact loss with different cutoffs for intra-chain and inter-chain contacts.
    """
    num_bins = dgram_logits.shape[-1]
    N = dgram_logits.shape[0]
    # all_bin_edges = bin_edges
    all_bin_edges = jnp.concatenate([bin_edges, jnp.array([jnp.inf])])  # [num_bins]

    is_intra = residue_chain[:, None] == residue_chain[None, :]  # True if same chain
    is_inter = ~is_intra

    cutoff_matrix = jnp.where(is_intra, intra_cutoff, inter_cutoff)  # [N, N]

    bin_edge_broadcast = all_bin_edges[None, None, :]  # [1, 1, num_bins]
    cutoff_broadcast = cutoff_matrix[:, :, None]       # [N, N, 1]

    mask = (bin_edge_broadcast < cutoff_broadcast).astype(jnp.bfloat16)  # [N, N, num_bins]

    q = jax.nn.softmax(dgram_logits, axis=-1)

    masked_logits = dgram_logits - 1e7 * (1.0 - mask)
    q_star = jax.nn.softmax(masked_logits, axis=-1)

    loss_matrix = -jnp.sum(q_star * jnp.log(q + 1e-8), axis=-1)  # [N, N]
    loss_matrix = loss_matrix * (1.0 - jnp.eye(N))

    # dont caculate the pad index residue_chain=0
    dis_mask = jnp.where(residue_chain[:, None] == 0, 0.0, 1.0)  # [N, 1]
    dis_mask = dis_mask * dis_mask.swapaxes(0, 1)  # [N, N]
    loss_matrix = loss_matrix * dis_mask
    loss = jnp.sum(loss_matrix) / (N * (N - 1))

    return loss

def contact_loss_dgram(
    dgram_logits: jnp.ndarray,        # [N, N, num_bins]
    bin_edges: jnp.ndarray,           # [num_bins - 1]
    residue_chain: jnp.ndarray,       # [N], 每个 residue 属于哪个链 (int)
    intra_cutoff: float = 14.0,
    inter_cutoff: float = 14.0,
    k_intra: int = 2,                # For intra-contact loss, take top k = 2
    k_inter: int = 1,                # For inter-contact loss, take top k = 1
    min_distance_intra: int = 9,     # Minimum distance for intra-contact (ignore contacts i−j < 9)
) -> jnp.ndarray:
    """
    Entropy-based contact loss with different cutoffs for intra-chain and inter-chain contacts.
    Additionally, computes the loss by considering only the top k contacts for each residue,
    and for intra-chain contacts, ignores those with i - j < 9.
    """
    num_bins = dgram_logits.shape[-1]
    N = dgram_logits.shape[0]

    # Adding an infinite value to the last bin edge for thresholding
    all_bin_edges = jnp.concatenate([bin_edges, jnp.array([jnp.inf])])  # [num_bins]

    # Determine intra-chain and inter-chain contacts
    is_intra = residue_chain[:, None] == residue_chain[None, :]  # True if same chain
    is_inter = ~is_intra

    # Apply cutoff values for intra and inter-chain distances
    cutoff_matrix = jnp.where(is_intra, intra_cutoff, inter_cutoff)  # [N, N]

    # Prepare for broadcasting
    bin_edge_broadcast = all_bin_edges[None, None, :]  # [1, 1, num_bins]
    cutoff_broadcast = cutoff_matrix[:, :, None]       # [N, N, 1]

    # Mask where the bin edges are below the cutoff
    mask = (bin_edge_broadcast < cutoff_broadcast).astype(jnp.bfloat16)  # [N, N, num_bins]

    # Apply softmax on the logits
    q = jax.nn.softmax(dgram_logits, axis=-1)

    # Mask the logits where no valid bin edges (above cutoff) and apply softmax
    masked_logits = dgram_logits - 1e7 * (1.0 - mask)
    q_star = jax.nn.softmax(masked_logits, axis=-1)

    # Compute the contact loss matrix
    loss_matrix = -jnp.sum(q_star * jnp.log(q + 1e-8), axis=-1)  # [N, N]
    loss_matrix = loss_matrix * (1.0 - jnp.eye(N))  # Exclude diagonal (self-contact)

    # Ignore residues that are marked with residue_chain=0 (pad index)
    dis_mask = jnp.where(residue_chain[:, None] == 0, 0.0, 1.0)  # [N, 1]
    dis_mask = dis_mask * dis_mask.swapaxes(0, 1)  # [N, N]
    loss_matrix = loss_matrix * dis_mask  # Mask out pad residues

    # For intra-chain contacts, exclude residue pairs where i - j < 9
    intra_distance = jnp.abs(jnp.arange(N)[:, None] - jnp.arange(N)[None, :])  # [N, N]
    loss_matrix = jnp.where(is_intra & (intra_distance < min_distance_intra), 0.0, loss_matrix)

    # For each residue, take the lowest k losses for intra and inter-chain contacts
    loss_per_residue = jnp.partition(loss_matrix, kth=k_inter, axis=-1)[:, :k_inter]  # [N, k_inter]
    if k_intra > 1:
        # For intra-chain contacts, we select the top k losses but for intra-chain we use k_intra
        loss_per_residue_intra = jnp.partition(loss_matrix * is_intra, kth=k_intra, axis=-1)[:, :k_intra]
        loss_per_residue = jnp.concatenate([loss_per_residue_intra, loss_per_residue], axis=-1)

    # Now compute the overall contact loss by averaging the top k contact losses
    loss = jnp.sum(loss_per_residue) / (N * (N - 1))

    return loss


def confidence_loss(predicted_lddt: jnp.ndarray, is_ligand: jnp.ndarray,example) -> float:
    ''' plddt_logits: [B,L,24,50]
        predicted_lddt: [B,L,24]
        is_ligand:[L,]'''
    # 先layout 去掉24 这一维度
    batch = feat_batch.Batch.from_data_dict(example) # feat_batch.Batch
    model_output_to_flat = atom_layout.compute_gather_idxs(
        source_layout=batch.convert_model_output.token_atoms_layout,
        target_layout=batch.convert_model_output.flat_output_layout,
    )
    predicted_lddt = predicted_lddt
    # model_output_to_flat = np.array(result['total_atom_nums'])
    if predicted_lddt is not None:
      pred_flat_b_factors = atom_layout.convert(
          gather_info=model_output_to_flat,
          arr=predicted_lddt,
          layout_axes=(-2, -1),
      )#[B,N] per atom

    #caculate the ligand plddt loss

    loss = 1-((jnp.sum(pred_flat_b_factors[0]) / len(is_ligand))/100)
    return loss

def af3_json(pdb_sequence:str, chain_id:list, name:str, seed:list, single:bool, ligandccd:list| None, ligand_id:list):
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
    af3_dic["bondedAtomPairs"] = None
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

  @functools.cached_property
  def model_params(self) -> hk.Params:
    """Loads model parameters from the model directory."""
    return params.get_model_haiku_params(model_dir=self._model_dir)

  # def debug_run(self)
  #   @hk.transform
  #   def forward_fn(batch):
  #     return model.Model(self._model_config)(batch)

  #   return functools.partial(
  #     forward_fn.apply, self.model_params
  #   )

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
        return model.Af3Design(self._model_config)(seq_logits,batch)

    return functools.partial(
        jax.jit(forward_fn.apply, device=self._device),
        self.model_params
    )


  def run_inference(
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
        batch = feat_batch.Batch.from_data_dict(featurised_example)
        seq_logits = jax.nn.one_hot(
              batch.token_features.aatype,
              residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP,
          )
        return seq_logits,featurised_example

    else:
        result = self._model(rng_key, featurised_example)

    # print(result['distogram']['bin_edges'].shape, result['distogram']['contact_probs'].shape) [63,] [256,256]

        result = jax.tree.map(np.asarray, result)
        result = jax.tree.map(
            lambda x: x.astype(jnp.float16) if x.dtype == jnp.bfloat16 else x,
            result,
        )
        result = dict(result)
        identifier = self.model_params['__meta__']['__identifier__'].tobytes()
        result['__identifier__'] = identifier
        return result

  def diffusion_result(self, result):
        result = jax.tree.map(np.asarray, result)
        result = jax.tree.map(
            lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x,
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


  def get_model(self,example):
      # forward pass
      @remat
      def _model(seq_logits, batch,rng):
        # logits -> sequence representation
        opt = batch['design_opt']
        seq_logits = self.updata_seq(seq_logits,opt)
        result=  self._designmodel(rng, seq_logits, batch)
        plddt_loss = confidence_loss(result['predicted_lddt'], batch['is_ligand'], example)
        dis_loss = contact_loss_dgram(result['distogram']['probs_logits'],
          result['distogram']['bin_edges'],
          batch['entity_id'])

        loss = dis_loss + plddt_loss
        return loss,result
      return jax.value_and_grad(_model, argnums=0,has_aux=True)

  def update_grad(self, grad, params, state):
    updates, new_state = self.o.update(grad, state, params)
    grad = jax.tree_util.tree_map(lambda x:-x, updates)
    return new_state, grad

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

  # seed: int
  inference_results: Sequence[model.InferenceResult]
  # full_fold_input: folding_input.Input
  embeddings: dict[str, np.ndarray] | None = None


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
    seq_logits,batch = model_runner.run_inference(example) # logits shape [L,31]
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
                #jax.tree_util.tree_map(lambda x: print(type(x), getattr(x, 'dtype', None)), (seq_logits, batch, example))

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
        # seq_logits[:mpnn_logits.shape[1],:mpnn_logits.shape[2]] = mpnn_logits[0]

        # print("stdout:", result.stdout)
        # print("stderr:", result.stderr)
        # print(new_seq)
        # print(logits.shape)



def update_example(seq_logits):
  #seq_logits (L,31) -> (L,21)
  seq_logits = seq_logits[:, :21]
  seq = argmax2seq(jnp.argmax(seq_logits, axis=-1))
  fold_input_dic = af3_json(pdb_sequence=seq, chain_id=['A'], name='test', seed=[42], single=True, ligandccd=['RET'], ligand_id=['B'])
  fold_inputs = folding_input.load_fold_inputs_from_path(fold_input_dic )
  for fold_input in fold_inputs:
    example =get_example(
        fold_input=fold_input,
        buckets=tuple(int(bucket) for bucket in _BUCKETS.value),
        conformer_max_iterations=_CONFORMER_MAX_ITERATIONS.value,
    )
  return example

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




def main(_):
  if _JAX_COMPILATION_CACHE_DIR.value is not None:
    jax.config.update(
        'jax_compilation_cache_dir', _JAX_COMPILATION_CACHE_DIR.value
    )

  if _JSON_PATH.value is None == _INPUT_DIR.value is None:
    raise ValueError(
        'Exactly one of --json_path or --input_dir must be specified.'
    )

  if not _RUN_INFERENCE.value and not _RUN_DATA_PIPELINE.value:
    raise ValueError(
        'At least one of --run_inference or --run_data_pipeline must be'
        ' set to true.'
    )

  if _INPUT_DIR.value is not None:
    fold_inputs = folding_input.load_fold_inputs_from_dir(
        pathlib.Path(_INPUT_DIR.value)
    )
  elif _JSON_PATH.value is not None:
    fold_inputs = folding_input.load_fold_inputs_from_path(
        pathlib.Path(_JSON_PATH.value)
    )
  else:
    raise AssertionError(
        'Exactly one of --json_path or --input_dir must be specified.'
    )

  # Make sure we can create the output directory before running anything.
  try:
    os.makedirs(_OUTPUT_DIR.value, exist_ok=True)
  except OSError as e:
    print(f'Failed to create output directory {_OUTPUT_DIR.value}: {e}')
    raise

  if _RUN_INFERENCE.value:
    # Fail early on incompatible devices, but only if we're running inference.
    gpu_devices = jax.local_devices(backend='gpu')
    if gpu_devices:
      compute_capability = float(
          gpu_devices[_GPU_DEVICE.value].compute_capability
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
        if _FLASH_ATTENTION_IMPLEMENTATION.value != 'xla':
          raise ValueError(
              'For devices with GPU compute capability 7.x (see'
              ' https://developer.nvidia.com/cuda-gpus) the'
              ' --flash_attention_implementation must be set to "xla".'
          )

  data_pipeline_config = None

  if _RUN_INFERENCE.value:
    devices = jax.local_devices(backend='gpu')
    print(
        f'Found local devices: {devices}, using device {_GPU_DEVICE.value}:'
        f' {devices[_GPU_DEVICE.value]}'
    )

    print('Building model from scratch...')
    model_runner = ModelRunner(
        config=make_model_config(
            flash_attention_implementation=typing.cast(
                attention.Implementation, _FLASH_ATTENTION_IMPLEMENTATION.value
            ),
            num_diffusion_samples=_NUM_DIFFUSION_SAMPLES.value,
            num_recycles=_NUM_RECYCLES.value,
            return_embeddings=_SAVE_EMBEDDINGS.value,
        ),
        device=devices[_GPU_DEVICE.value],
        model_dir=pathlib.Path(MODEL_DIR.value),
    )
    # Check we can load the model parameters before launching anything.
    print('Checking that model parameters can be loaded...')
    _ = model_runner.model_params
  else:
    model_runner = None

  length = 99
  ligand_ccd = 'RET'

  seq = argmax2seq(np.zeros((length,), dtype=np.int32))
  fold_input_dic = af3_json(pdb_sequence=seq, chain_id=['A'], name='test', seed=[42], single=True, ligandccd=[ligand_ccd], ligand_id=['B'])
  fold_inputs = folding_input.load_fold_inputs_from_path(fold_input_dic ) #可以接受输入的是json path也可以直接是dic #pathlib.Path(_JSON_PATH.value)
  for fold_input in fold_inputs:
    if _NUM_SEEDS.value is not None:
      print(f'Expanding fold job {fold_input.name} to {_NUM_SEEDS.value} seeds')
      fold_input = fold_input.with_multiple_seeds(_NUM_SEEDS.value)
    example = get_example(
        fold_input=fold_input,
        buckets=tuple(int(bucket) for bucket in _BUCKETS.value),
        conformer_max_iterations=_CONFORMER_MAX_ITERATIONS.value,
    )
    grad = design(model_runner,example)



if __name__ == '__main__':
  # flags.mark_flags_as_required(['output_dir'])
  app.run(main)

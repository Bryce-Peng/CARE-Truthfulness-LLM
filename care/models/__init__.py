# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING

from transformers.utils import (
                                OptionalDependencyNotAvailable,
                                _LazyModule,
                                is_flax_available,
                                is_tf_available,
                                is_torch_available,
                                is_tokenizers_available,
                            )
from transformers.utils.import_utils import define_import_structure

_import_structure = {
    #"configuration_mistral": ["MistralConfig"],
    #"configuration_qwen2": ["Qwen2Config"],
    #"tokenization_qwen2": ["Qwen2Tokenizer"],
}


try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_mistral"] = set([
        "MistralForCausalLM",
        "MistralForQuestionAnswering",
        "MistralModel",
        "MistralPreTrainedModel",
        "MistralForSequenceClassification",
        "MistralForTokenClassification",
    ])
    _import_structure["modeling_qwen2"] = set([
        "Qwen2ForCausalLM",
        "Qwen2ForQuestionAnswering",
        "Qwen2Model",
        "Qwen2PreTrainedModel",
        "Qwen2ForSequenceClassification",
        "Qwen2ForTokenClassification",
    ])
    _import_structure["modeling_phi"] = [
        "PhiPreTrainedModel",
        "PhiModel",
        "PhiForCausalLM",
        "PhiForSequenceClassification",
        "PhiForTokenClassification",
    ]
    

if TYPE_CHECKING:
    from .modeling_llama import *
    from .modeling_gemma import *
    from .modeling_glm import *
    from .modeling_gptj import *
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_mistral import (
            MistralForCausalLM,
            MistralForQuestionAnswering,
            MistralForSequenceClassification,
            MistralForTokenClassification,
            MistralModel,
            MistralPreTrainedModel,
        )
    from .modeling_phi import *
    from .modeling_phi3 import *
    from .modeling_qwen2 import *
    
else:
    import sys

    _file = globals()["__file__"]
    dynamic_structure = define_import_structure(_file)
    
    combined_structure = {frozenset({'torch'}): {**dynamic_structure[frozenset({'torch'})], **_import_structure}}
    
    sys.modules[__name__] = _LazyModule(
        __name__,
        _file,
        combined_structure,  
        module_spec=__spec__
    )
    #sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
    #sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)


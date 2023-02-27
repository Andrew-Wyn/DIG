import os
import torch

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    # DataCollatorWithPadding,
    # EvalPrediction,
    # HfArgumentParser,
    # PretrainedConfig,
    # Trainer,
    # TrainingArguments,
    # default_data_collator,
)

# TODO: capire perche se non setto cache_dir in AutoTokenizer
# non usa come cache la directory specificata
CACHE_DIR = f"{os.getcwd()}/.hf_cache/"
# change Transformer cache variable
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TaskVector():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                pretrained_state_dict = self._load_from_hf(pretrained_checkpoint) # the pretrained is taken from hf
                finetuned_state_dict = torch.load(finetuned_checkpoint)
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
    
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_model = torch.load(pretrained_checkpoint)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                pretrained_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]

        return pretrained_model

    def save_vector_in_hf_model(self, model, vector):
        for module_ in model.named_modules():
            if f"{module_[0]}.weight" in vector.keys() or f"{module_[0]}.bias" in vector.keys():
                module_[1].weight.data = vector[f"{module_[0]}.weight"]
                
                try:
                    module_[1].bias.data = vector[f"{module_[0]}.bias"]
                except KeyError as e:
                    print(e)

        return model
    
    def _load_from_hf(self, model):
        model_weights = dict()

        for module_ in model.named_modules():
            pass

        return model_weights

if __name__ == "__main__":
    pass

    """
    
        1) Take MECO Finetuned Model (EN ALL) and generate task vector v2
        2) Add v2 to SST2 Finetuned Model
        3) Save resulting model in .bin format

    pretrained_checkpoint = 

    task_vector = TaskVector(pretrained_checkpoint, pretrained_checkpoint)

    pretrained_checkpoint = "finetuning/notpretraining/finetuning_dir_it_all_notpretraining/model-prajjwal1/bert-tiny-finetuned-randomized-full/pytorch_model.bin"
    task_vector = TaskVector(pretrained_checkpoint, pretrained_checkpoint)

    print(task_vector.vector.keys())

    a = task_vector.apply_to(pretrained_checkpoint)

    #print(a)

    new_model_weights = task_vector.vector

    model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", cache_dir=CACHE_DIR)



    for module_ in model.named_modules():
        try:
            print(module_[0])
            print(module_[1].weight.data)
            print(module_[1].bias.data)
            print()
        except AttributeError as e:
            print(e)

    print(model)
    """
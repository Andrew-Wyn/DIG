import os
import torch
import argparse

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
                pretrained_state_dict = torch.load(pretrained_checkpoint)
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
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='taskVector')
    parser.add_argument('-hfsst2model', type=str)
    parser.add_argument('-pretrainedmodel',  type=str) # specify the .bin file inside hf_cache dir
    parser.add_argument('-finetunedmodelmeco',  type=str) # specify the .bin file inside finetuned dir
    parser.add_argument('-finetunedmodelsst2', type=str)

    args = parser.parse_args()

    task_vector_meco = TaskVector(args.pretrainedmodel, args.finetunedmodelmeco)

    sst2_meco_model = task_vector_meco.apply_to(args.finetunedmodelsst2)

    config = AutoConfig.from_pretrained(args.hfsst2model)

    torch.save(sst2_meco_model, "tv_tmp.bin")

    model = AutoModelForSequenceClassification.from_pretrained("tv_tmp.bin", config=config)

    print(model)

    print("SEE THE WEIGHTS...")

    for module_ in model.named_modules():
        try:
            print(module_[0])
            print(module_[1].weight.data)
            print(module_[1].bias.data)
        except AttributeError as e:
            print(e)

    model.save_pretrained("sst2_meco_model")
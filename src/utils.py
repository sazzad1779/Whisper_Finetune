# STEP 1. Download Dataset
from datasets import load_dataset, DatasetDict
from huggingface_hub import  HfFolder
### https://github.com/vb100/whisper_ai_finetune/blob/main/whisper_finetuning.py#L100
class Utils:
    def __init__(self):
        self.dataset_path = None 
    def load_dataset(self, dataset_path,  language:str=None, split:str=None, trust_remote_code=True):
        """
        Load a specific version of the Common Voice dataset.

        Args:
            dataset_path(str): directory of dataset
            language (str): Language code for the dataset (e.g., "lt", "bn").
            split (str): Dataset split to load (e.g., "train", "validation", "test").
            trust_remote_code (bool): Whether to trust remote code during loading.

        Returns:
            Dataset: The requested dataset split.
        """
        return load_dataset(
            dataset_path, language, split=split, trust_remote_code=trust_remote_code
        )
    # import the relavant libraries for loggin in

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Functions and procedures
    def save_hf_token(self,token):
        """
        Save the Hugging Face token to the default location.

        Args:
            token (str): The Hugging Face token to save.
        """
        try:
            HfFolder.save_token(token)
            print("Token saved successfully.")
        except Exception as e:
            print(f"Error saving token: {e}")
        




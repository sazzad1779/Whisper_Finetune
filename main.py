from src.utils import Utils
from datasets import DatasetDict

if __name__== "__main__":
    utils = Utils()
    token =input("Enter your Huggingface token: ")
    utils.save_hf_token(token)
    # Create a DatasetDict to store splits
    common_voice = DatasetDict()
    dataset =utils.load_dataset("mozilla-foundation/common_voice_17_0","bn","test[:1]")
    print(dataset)



from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader


def get_c4_dataloader(data_path, tokenizer, max_length, batch_size):

    data = load_dataset(data_path, streaming=True)['train']
    def tokenizer_func(data):
        output = tokenizer(data['text'], 
                            truncation=True,
                            max_length=max_length,
                            padding="max_length")
        return output
    dataset = data.map(tokenizer_func, remove_columns=["url", "timestamp", "text"])
    collator_fun = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator_fun)
    return dataloader

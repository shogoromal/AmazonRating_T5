
#ベースコード
#https://github.com/kevinscaria/InstructABSA/blob/main/InstructABSA/utils.py

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import (
    DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Trainer, Seq2SeqTrainer
)

class Config(object):
    """Wrapper class for model hyperparameters."""
    def __init__(self, args=None):
        """
        Defaults
        """
        self.model = args.get('model', None)
        self.model_checkpoint = args.get('model_checkpoint', None)
        self.inst_type = args.get('inst_type', None)
        self.experiment_name = args.get('experiment_name', None)
        self.task = args.get('task', None)
        self.output_dir = args.get('output_dir', None)
        self.id_tr_data_path = args.get('id_tr_data_path', None)
        self.id_te_data_path = args.get('id_te_data_path', None)
        self.id_val_data_path = args.get('id_val_data_path', None)
        self.set_instruction_key = args.get('set_instruction_key', 1)
        self.ood_tr_data_path = args.get('ood_tr_data_path', None)
        self.ood_te_data_path = args.get('ood_te_data_path', None)
        self.output_path = args.get('output_path', None)
        self.sample_size = args.get('sample_size', 1)
        self.evaluation_strategy = args.get('evaluation_strategy', 'no')
        self.learning_rate = args.get('learning_rate', 5e-5)
        self.per_device_train_batch_size = args.get('per_device_train_batch_size', 32)
        self.per_device_eval_batch_size = args.get('per_device_eval_batch_size', 16)
        self.num_train_epochs = args.get('num_train_epochs', 4)
        self.weight_decay = args.get('weight_decay', 0.01)
        self.warmup_ratio = args.get('warmup_ratio', 0.1)
        self.save_strategy = args.get('save_strategy', 'no')
        self.load_best_model_at_end = args.get('load_best_model_at_end', False)
        self.push_to_hub = args.get('push_to_hub', False)
        self.eval_accumulation_steps = args.get('eval_accumulation_steps', 1)
        self.predict_with_generate = args.get('predict_with_generate', True)
        self.max_token_length = args.get('max_token_length', 1024)
        self.bos_instruction = args.get('bos_instruction', None)
        self.delim_instruction = args.get('delim_instruction', None)
        self.eos_instruction = args.get('eos_instruction', None)
        self.test_input = args.get('test_input', None)


class T5Classifier:
    def __init__(self, model_checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, force_download = True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, force_download = True)
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer)
        self.device = 'cuda' if torch.has_cuda else ('mps' if torch.has_mps else 'cpu')

    def tokenize_function_inputs(self, sample):
        """
        Udf to tokenize the input dataset.
        """
        sample['input_ids'] = self.tokenizer(sample["instruct_text"], max_length = 1024, truncation = True).input_ids
        sample['labels'] = self.tokenizer(sample["labels"], max_length = 64, truncation = True).input_ids
        return sample

    def train(self, train_tk_ds, test_tk_ds, **kwargs):
        """
        Train the generative model.
        """

        # Set training arguments
        args = Seq2SeqTrainingArguments(
            **kwargs
            )

        # Define trainer object
        trainer = Trainer(
            model = self.model,
            args = args,
            train_dataset=train_tk_ds,
            eval_dataset=test_tk_ds,
            tokenizer=self.tokenizer,
            data_collator = self.data_collator
        )
        print("Trainer device:", trainer.args.device)

        # Finetune the model
        torch.cuda.empty_cache()
        print('\nModel training started ....')
        trainer.train()

        # Save best model
        trainer.save_model()
        return trainer

    def get_labels(self, tokenized_dataset, batch_size = 4):
        """
        Get the predictions from the trained model.
        """
        def collate_fn(batch):
            input_ids = [torch.tensor(example['input_ids']) for example in batch]
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            return input_ids

        dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=collate_fn)
        predicted_output = []
        self.model.to(self.device)
        print('Model loaded to: ', self.device)

        for batch in tqdm(dataloader):
            batch = batch.to(self.device)
            output_ids = self.model.generate(batch)
            output_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for output_text in output_texts:
                predicted_output.append(output_text)
        return predicted_output

    def get_metrics(self, y_true, y_pred):
        return precision_score(y_true, y_pred, average='macro'), recall_score(y_true, y_pred, average='macro'), \
            f1_score(y_true, y_pred, average='macro'), accuracy_score(y_true, y_pred)
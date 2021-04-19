import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, batch_to_device, beam_search, sample_sequence
from tqdm  import tqdm
from transformers import GPT2Tokenizer

import json

SPECIAL_TOKENS = ["<bos>", "<eos>", "<que>", "<ans>", "<speaker>", "<subtitle>",
                  "<bounding_feature>", "<person>", "<behavior>", "<emotion>", "<video>", "<pad>"]
SPECIAL_TOKENS_DICT = {'bos_token': "<bos>", 'eos_token': "<eos>", 'additional_special_tokens': ["<que>", "<ans>", "<speaker>", "<subtitle>", "<bounding_feature>", "<person>", "<behavior>", "<emotion>", "<video>"], 'pad_token': "<pad>"}

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config['trainer']
        self.model_inputs = config['data_loader']['args']['inputs']
        self.generator = config['generator']
        self.data_loader = data_loader
        self.video = config['data_loader']['args']['video']
        self.bbfts = config['data_loader']['args']['bbfts']
        
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        tokenizer_class = GPT2Tokenizer
        self.tokenizer = tokenizer_class.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        tqdm_bar = tqdm(self.data_loader, desc='Train Epoch : {}'.format(epoch))
        gradient_accumulation_steps = 8
        iteration = 0
        max_norm = 1.0
        for batch_idx, batch in enumerate(tqdm_bar):
            input_ids = batch[0].to(self.device)
            token_type_ids = batch[1].to(self.device)
            lm_labels = batch[2].to(self.device)
            answer_list = batch[3]
            input_mask = batch[4].to(self.device)
            input_ids = self.model.transformer.wte(input_ids)

            # processing bounding features
            if self.bbfts:
                bbfts_list = batch[5][0].to(self.device)
                bbfts = self.model.align_model(bbfts_list.float()).unsqueeze(0)
                input_ids = torch.cat([bbfts, input_ids], dim = 1)
                token_type_ids = torch.cat([torch.ones((bbfts.size(0), bbfts.size(1))).long().cuda() * self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[6]), token_type_ids], dim = 1) 

            if self.video: 
                i3d = batch[6][0].to(self.device)
                i3d = i3d.unsqueeze(0)
                video_mask = batch[7].to(self.device)
                reply_mask = batch[8].to(self.device)
                video_embs = self.model.video_ff(i3d) 
                input_embs = torch.cat([video_embs, input_ids], dim=1)
                token_type_ids = torch.cat([torch.ones((i3d.size(0), i3d.size(1))).long().cuda() * self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-2]), token_type_ids], dim=1)

                if self.bbfts:
                    bbfts = self.model.video_inverse_ff(bbfts)
                    i3d = torch.cat([i3d, bbfts], dim = 1) 
                
                video_loss = self.model(input_embs, token_type_ids = token_type_ids, labels=(lm_labels, i3d), attention_mask=[video_mask, input_mask], mode = "video")[0]
                reply_loss = self.model(input_embs, token_type_ids = token_type_ids, labels=(lm_labels, i3d), attention_mask=[reply_mask, input_mask], mode = "reply")[0]
                loss = (video_loss + reply_loss) / gradient_accumulation_steps
            else:
                loss = self.model(input_embs = input_ids, token_type_ids = token_type_ids, labels = lm_labels)[0]
        
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            if iteration % gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            iteration = iteration + 1

#            for met in self.metric_ftns:
#                if 'accuracy_diff' in met.__name__:
#                    self.train_metrics.update(met.__name__, *met(output, target, batch['q_level_logic']))
#                else:
#                    self.train_metrics.update(met.__name__, met(answer_list, target, self.model.vocab))


            if batch_idx % self.log_step == 0 or batch_idx == self.len_epoch - 1:
                tqdm_bar.set_description('Train Epoch: {} {} Loss: {:.6f}'.format(
                                         epoch,
                                         self._progress(batch_idx),
                                         loss.item()))
                """
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                """

                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()
    

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+ k : v for k, v in val_log.items()})

        self.tokenizer.save_vocabulary('log/')
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            tqdm_bar = tqdm(self.valid_data_loader, desc='Valid Epoch: {}'.format(epoch))
            for batch_idx, batch in enumerate(tqdm_bar):
                input_ids = batch[0].to(self.device)
                token_type_ids = batch[1].to(self.device)
                target = batch[3]
                i3d = None
                bbfts = None
                if self.bbfts:
                    bbfts_list = batch[5][0].to(self.device)
                    bbfts = self.model.align_model(bbfts_list.float()).unsqueeze(0)
                if self.video:
                    i3d = batch[6].to(self.device)

                self.optimizer.zero_grad()
                # deprecated
                if self.generator['is_beam_search']:
                    output = beam_search(self.model, input_ids, token_type_ids, self.tokenizer, self.device, video=i3d)[0][0]
                else:                
                    output = sample_sequence(self.model, input_ids, token_type_ids, self.tokenizer, self.device, video=i3d, bbfts = bbfts)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                for met in self.metric_ftns:
                    if 'accuracy_diff' in met.__name__:
                        self.valid_metrics.update(met.__name__, *met(output, target, batch['q_level_logic']))
                    else:
                        self.valid_metrics.update(met.__name__, met(output, target[0], self.tokenizer))


                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        if self.config['add_histogram']:
            for name, p in self.model.named_parameters():
                self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            total = self.data_loader.n_samples
            current = min((batch_idx + 1) * self.data_loader.batch_size, total)
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


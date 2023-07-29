from transformers import LlamaForCausalLM
import torch.nn as nn


class CustomLlamaModel(LlamaForCausalLM):

    def extract_weight_positions(self, input_ids):

    def get_weights(self, input_ids, weight_size=2.0):
        # Use your method to extract the weight positions
        weight_positions = self.extract_weight_positions(input_ids)

        # Create a tensor of weights, where the weight is 2.0 at the weight positions and 1 elsewhere
        weights = torch.where(weight_positions, weight_size, 1.0)

        # Move the weights to the same device as the input_ids
        weights = weights.to(input_ids.device)

        return weights

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            # Get the weights
            weights = self.get_weights(input_ids[..., 1:]).view(-1)

            # Compute the loss
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(shift_logits, shift_labels)

            # Apply weights
            weighted_losses = losses * weights

            # Compute the mean loss
            loss = weighted_losses.mean()

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

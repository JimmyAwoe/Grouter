import torch
import torch.nn.functional as F
import torch.distributed as dist
import warnings

_WARNED = False

def logits_hook_utils_func(logits_dict, logits_length, logits_device, model_tag=None, transpose=False):
    @torch.no_grad()
    def fetch_logits_hook(name):
        logits_dict[name] = torch.empty(logits_length, device=logits_device)
        def hook(module, input, output):
            global _WARNED
            hidden_state = input[0]
            h = hidden_state.shape[-1]       
            
            if transpose:
                hidden_state = hidden_state.permute(1, 0, 2).contiguous()
            if dist.is_initialized() and dist.get_rank() == 0:
                if not _WARNED:
                    warnings.warn(f"Please check the seq len is {hidden_state.shape[-3]} and "
                                    f"bsz is {hidden_state.shape[-2]}", UserWarning)
                    _WARNED = True
            else: 
                if not _WARNED:
                    warnings.warn(f"Please check the seq len is {hidden_state.shape[-3]} and "
                                    f"bsz is {hidden_state.shape[-2]}", UserWarning)
                    _WARNED = True
            ### compute gating score
            logits = F.linear(hidden_state.view(-1, h), module.weight, None)
            logits_dict[name] = logits.to(logits_device)
        return hook

    def fetch_logits_hook_hy(name):
        logits_dict[name] = torch.empty(logits_length, device=logits_device)
        @torch.no_grad()
        def hook(module, input, output):
            # FIXME add shape verification
            _, _, hidden_size = input[0].shape
            hidden_states = input[0].reshape(-1, hidden_size)
            if module.wg.weight.dtype == torch.float32:
                hidden_states = hidden_states.float()
            logits = module.wg(hidden_states)
            logits = logits.float()
            logits_dict[name] = logits.to(logits_device)
        return hook
    if model_tag == 'hunyuan':
        return fetch_logits_hook_hy
    else:
        return fetch_logits_hook
        

def fetch_start_event_hook_auxiliary_fun(event_list: dict):
    """import event_list to help monitor the process"""
    def fetch_start_event_hook(name):
        """fetch a hook to initialize cuda start adn end event"""
        def hook(module, input, output):
            torch.cuda.synchronize()
            dist.barrier()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)            
            event_list[f'{name}_start'] = start_event
            event_list[f'{name}_end'] = end_event
            start_event.record()
        return hook
    return fetch_start_event_hook

def fetch_end_event_hook_auxiliary_fun(event_list: dict):
    """import_event list to help monitor the process"""
    def fetch_end_event_hook(name):
        """fetch a hook to end the timing process and record time usage"""
        event_list[f'{name}_time_usage'] = None
        def hook(module, input, output):
            end_event = event_list[f'{name}_end']
            start_event = event_list[f'{name}_start']
            end_event.record()
            torch.cuda.synchronize()
            use_time = start_event.elapsed_time(end_event)
            use_time = torch.tensor(use_time).to(next(module.parameters()).device)
            dist.all_reduce(use_time, op=dist.ReduceOp.AVG)
            if event_list[f'{name}_time_usage'] == None:
                event_list[f'{name}_time_usage'] = use_time.item()
            else:
                event_list[f'{name}_time_usage'] += use_time.item()
                
        return hook
    return fetch_end_event_hook
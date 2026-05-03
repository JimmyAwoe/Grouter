import torch
def fetch_grouter_logits_hook(name, grouter_indices_dict):
    def grouter_hook(module, input, output):
        """This is the hook specialized for TopKGrouter. It finish the last step for routing"""
        if isinstance(grouter_indices_dict[name], tuple):
            topk_idx, logits = grouter_indices_dict[name]
            if isinstance(topk_idx, tuple):
                topk_idx = tuple(idx.detach().clone() for idx in topk_idx)
                scores, routing_map = module.routing(logits.detach().clone(), topk_idx)
            else:
                scores, routing_map = module.routing(logits.detach().clone(), topk_idx.detach().clone())
        else:
            scores, routing_map = module.routing(output, grouter_indices_dict[name].detach().clone())
        return scores, routing_map
    return grouter_hook

def fetch_grouter_scores_hook(name, grouter_indices_dict):
    def grouter_hook(module, input, output):
        """This is the hook specialized for TopKGrouter. It finish the last step for routing"""
        scores, routing_map = grouter_indices_dict[name]
        return scores, routing_map
    return grouter_hook
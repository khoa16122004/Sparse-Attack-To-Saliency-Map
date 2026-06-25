import torch

class MarginSalinecy_Fitness:
    def __init__(self, model, x_tensor, normalize, y_true, explain_method):
        self.model = model
        self.y_true = y_true
        self.normalize = normalize
        self.explain_method = explain_method
        # print(x_tensor.shape)
        self.saliency_true, _ = self.explain_method(self.model, x_tensor, self.normalize,  self.y_true)
        # print("Saliency map shape: ", self.saliency_true.shape)
        
        
    def benchmark(self, xadv_tensors):
        saliency_maps, logits = self.explain_method(self.model, xadv_tensors, self.normalize, self.y_true)
        margin_loss = self.cal_marginloss(logits, self.y_true)
        saliency_loss = self.cal_saliency_loss(saliency_maps, self.saliency_true)
        return margin_loss, saliency_loss, logits
    
        
    def cal_marginloss(self, logits, y_true):
        if y_true.numel() == 1:
            y_true = y_true.expand(logits.size(0))
        true_logits = logits.gather(1, y_true.unsqueeze(1)).squeeze(1)
        tmp = logits.clone()
        tmp.scatter_(1, y_true.unsqueeze(1), float("-inf"))
        max_other_logits = tmp.max(dim=1).values
        margin = true_logits - max_other_logits

        return margin
    
    
    def cal_saliency_loss(self, saliency_maps, saliency_true, eps=1e-12):
        saliency_maps = saliency_maps.flatten(start_dim=1)
        saliency_true = saliency_true.flatten(start_dim=1)
        if saliency_true.size(0) == 1 and saliency_maps.size(0) > 1:
            saliency_true = saliency_true.expand(saliency_maps.size(0), -1)
        inter = torch.minimum(saliency_maps, saliency_true).sum(dim=1)
        union = torch.maximum(saliency_maps, saliency_true).sum(dim=1)
        soft_iou = inter / (union + eps)
        return soft_iou
            
            
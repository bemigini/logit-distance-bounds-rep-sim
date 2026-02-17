import torch
import torch.nn.functional as F


def Classification_Loss(out_dict: dict, args):
    out = out_dict['LOGITS']
    labels = out_dict['LABELS'].to(torch.long)


    if args.model in ['classifier', 'logitdistill']:
        loss = F.cross_entropy(out, labels, reduction='mean')
    else:
        return NotImplementedError('Choice not implemented')
        
    assert loss > 0, loss
    
    losses = {'y-loss': loss.item()}        

    return loss, losses

def BiClassification_Loss(out_dict: dict, args):
    out = out_dict['LOGITS']
    labels = out_dict['LABELS'].to(torch.long)
    out_concepts = out_dict['CLOGITS']
    concepts = out_dict['TLOGITS'].to(torch.long)

    loss = F.cross_entropy(out, labels, reduction='mean')
    assert loss > 0, loss
    losses = {'y-loss': loss.item()}     

    loss2 = F.cross_entropy(out_concepts, concepts, reduction='mean')
    assert loss2 > 0, loss2
    losses.update({'c-loss': loss2.item()})   

    return loss + loss2, losses

def L1_Loss_Logits(out_dict: dict, args):
    """
    Calculates the L1 logit loss
    
    :param out_dict: Dictionary for saving the losses
    :type out_dict: dict
    :param args: Other run arguments
    """
    student_logits = out_dict['LOGITS']
    teacher_logits = out_dict['TLOGITS']

    final_loss = F.l1_loss(student_logits, teacher_logits, reduction='mean')
    assert final_loss >= 0, final_loss

    kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    log_p_target = F.log_softmax(teacher_logits, dim=1)
    log_p_student = F.log_softmax(student_logits, dim=1)
    kl_div = kl_loss(log_p_student, log_p_target)

    losses = {'mse-logits-loss': final_loss.item()}
    losses.update({'kl-loss': kl_div.item()})

    return final_loss, losses


def KL_Loss_logits(out_dict: dict, args):
    """
    Calculates the KL divergence loss
    
    :param out_dict: Dictionary for saving the losses
    :type out_dict: dict
    :param args: Other run arguments
    """
    student_logits = out_dict['LOGITS']
    teacher_logits = out_dict['TLOGITS']

    kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    log_p_target = F.log_softmax(teacher_logits, dim=1)
    log_p_student = F.log_softmax(student_logits, dim=1)
    final_loss = kl_loss(log_p_student, log_p_target)
    assert final_loss >= 0, final_loss

    losses = {'kl-loss': final_loss.item()}

    return final_loss, losses


def L2_Loss_Logits(out_dict: dict, args):
    """
    Calculates the squared L2 logit loss
    
    :param out_dict: Dictionary for saving the losses
    :type out_dict: dict
    :param args: Other run arguments
    """
    student_logits = out_dict['LOGITS']
    teacher_logits = out_dict['TLOGITS']
    num_inputs = student_logits.shape[0]

    squared_errors = torch.pow(student_logits - teacher_logits, 2)
    final_loss = torch.sum(squared_errors) / num_inputs
    assert final_loss >= 0, final_loss

    kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    log_p_target = F.log_softmax(teacher_logits, dim=1)
    log_p_student = F.log_softmax(student_logits, dim=1)
    kl_div = kl_loss(log_p_student, log_p_target)

    losses = {'mse-logits-loss': final_loss.item()}
    losses.update({'kl-loss': kl_div.item()})

    return final_loss, losses

import torch

def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'pwfharmonization':
        assert(opt.dataset_mode == 'ADNI')
        from .pwfharmonization_model import UnsupModel
        model = UnsupModel()


        # print("Total number of paramerters in networks is {}  ".format(sum(x.nelement() for x in model.parameters())))
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model



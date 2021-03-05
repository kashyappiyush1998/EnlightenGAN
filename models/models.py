from .single_model import SingleModel

def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'single':
        # assert(opt.dataset_mode == 'unaligned')
        model = SingleModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    print("Initializing Models")
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model

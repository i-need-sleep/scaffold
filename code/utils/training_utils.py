def load_checkpoint_if_available(Model, args):
    if args.checkpoint is not None:
        print(f'=====\nLoading checkpoint from {args.checkpoint}\n=====\n')
        return Model.load_from_checkpoint(args.checkpoint, vars(args), args=vars(args))
    else:
        return Model(vars(args))
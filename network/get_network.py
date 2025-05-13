from network import ResNet



def GetNetwork(args, num_classes, pretrained=True, **kwargs):
    if args.model == 'resnet18':
        model = ResNet.resnet18(pretrained=pretrained, num_classes=num_classes, **kwargs)
        feature_level = 512
        
    elif args.model == 'resnet18_rsc':
        model = ResNet.resnet18_rsc(pretrained=pretrained, num_classes=num_classes, **kwargs)
        feature_level = 512

    elif args.model == 'resnet50':
        model = ResNet.resnet50(pretrained=pretrained, num_classes=num_classes, **kwargs)
        feature_level = 2048
        
    elif args.model == 'resnet50_rsc':
        model = ResNet.resnet50_rsc(pretrained=pretrained, num_classes=num_classes, **kwargs)
        feature_level = 2048

    else:
        raise ValueError("The model is not support")

    return model, feature_level

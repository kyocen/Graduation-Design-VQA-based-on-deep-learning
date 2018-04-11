def predict(val_loader, model):
    model.eval()#调整到eval模式
    itoa = val_loader.dataset.codebook['itoa']

    results = []
    bar = progressbar.ProgressBar()
    # sample: (que_id, img, que, [obj])
    for sample in bar(val_loader):
        #[question_id,ndarray(image feature),question:[list of word index],ndarray(object feature),answers(两种模式)]
        sample_var = [Variable(d).cuda() for d in list(sample)[1:]]

        score = model(*sample_var)#2d variable (batch_size, 3000 1d score vector)

        results.extend(format_result(sample[0], score, itoa))

    return results#list of dict{'question_id': que_id,'answer': model's answer} (batch x batch_size) x dict
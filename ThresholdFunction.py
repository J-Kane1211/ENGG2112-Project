def threshold(modelOutput):
    if modelOutput > 0.5:
        return 1
    else:
        return 0
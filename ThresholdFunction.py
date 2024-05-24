def threshold(modelOutput):
    if modelOutput < 0.1:
        print("Low risk. No need to address.")
    elif (modelOutput >= 0.1 and modelOutput < 0.35):
        print("Medium risk. Address at next GP appointment.")
    elif (modelOutput >= 0.35 and modelOutput < 0.75):
        print("High risk. Book appointment within the next 2 months.")
    else:
        print("Very high risk. Book appointment ASAP.")
        

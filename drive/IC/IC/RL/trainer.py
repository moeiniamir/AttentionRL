n_classes = 80


with torch.no_grad():
    max_seq_len = 17
    captions, features, urls = sample_coco_minibatch(small_data, batch_size=100, split='val')
    for i in range(100):
        gen_caps = []
        gen_caps.append(GenerateCaptions(features[i:i+1], captions[i:i+1], policyNet)[0])
        gen_caps.append(GenerateCaptionsWithBeamSearch(features[i:i+1], captions[i:i+1], policyNet)[0][0][0])
        gen_caps.append(GenerateCaptionsWithBeamSearchValueScoring(features[i:i+1], captions[i:i+1], policyNet)[0][0][0])
        decoded_tru_caps = decode_captions(captions[i], data["idx_to_word"])

#         f = open("truth3.txt", "a")
#         f.write(decoded_tru_caps + "\n")
#         f.close()
        
#         decoded_gen_caps = decode_captions(gen_caps[0], data["idx_to_word"])
#         f = open("greedy3.txt", "a")
#         f.write(decoded_gen_caps + "\n")
#         f.close()
        
#         decoded_gen_caps = decode_captions(gen_caps[1], data["idx_to_word"])
#         f = open("beam3.txt", "a")
#         f.write(decoded_gen_caps + "\n")
#         f.close()
        
#         decoded_gen_caps = decode_captions(gen_caps[2], data["idx_to_word"])
#         f = open("policyvalue3.txt", "a")
#         f.write(decoded_gen_caps + "\n")
#         f.close()
        try:
            plt.imshow(image_from_url(urls[i]))
            plt.show()
        except:
            continue
        print(urls[i])
        print(decode_captions(gen_caps[2], data["idx_to_word"]))

        
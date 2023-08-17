def get_words_in_time_range(word_list, start_time, end_time):
    words = []

    for word in word_list:
        word_s, word_e, text = float(word[0]), float(word[1]), word[2]

        if word_s >= end_time:
            break

        if word_e <= start_time:
            continue

        words.append(text)

    return words
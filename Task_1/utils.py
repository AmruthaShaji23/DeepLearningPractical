import pandas as pd
import random


def read_ngrams():  # read each line, and see how many characters is in the line, and seperate (read the underscores)
    output_grams = [[], [], [], [], [], [], [], [], []]  # found that the max gram is 10
    # read csv
    df = pd.read_csv('ngrams_frequencies_withNames.csv')
    grams = df['Names']
    frequencies = df['Frequencies']
    for i, gram in enumerate(grams):
        count = gram.count('_')  # counting the underscores

        def exception():
            print("Value is something else")

        gram_reversed = "_".join(reversed(gram.split('_')))  # reverse the gram
        switch = {
            1: lambda: output_grams[0].append((gram_reversed, frequencies[i])),  # two characters, one underscore
            2: lambda: output_grams[1].append((gram_reversed, frequencies[i])),  # three characters, two underscores
            3: lambda: output_grams[2].append((gram_reversed, frequencies[i])),  # four charactes, three underscores
            4: lambda: output_grams[3].append((gram_reversed, frequencies[i])),  # etc
            5: lambda: output_grams[4].append((gram_reversed, frequencies[i])),
            6: lambda: output_grams[5].append((gram_reversed, frequencies[i])),
            7: lambda: output_grams[6].append((gram_reversed, frequencies[i])),
            8: lambda: output_grams[7].append((gram_reversed, frequencies[i])),
            9: lambda: output_grams[8].append((gram_reversed, frequencies[i]))
        }

        switch.get(count, exception)()

    return output_grams  # its already ordered for frequency per line bc of how we read and append


def synthetic_line(n_grams):  # Making the synthetic line

    # we can change these,
    min_word_len = 2
    max_word_len = 6
    min_line_len = 6
    max_line_len = 40

    line = []
    total_chars = 0
    target_line_len = random.randint(min_line_len, max_line_len)

    while total_chars < target_line_len:
        word = []
        word_char_len = random.randint(min_word_len, max_word_len)
        word_chars = 0

        while word_chars < word_char_len:
            valid_choices = []

            for n, group in enumerate(n_grams, start=2):
                if group:
                    # filter n-grams that can still fit in the current word
                    filtered = [(gram, freq) for gram, freq in group if n + word_chars <= word_char_len]
                    if filtered:
                        grams, freqs = zip(*filtered)
                        valid_choices.append((n, grams, freqs))

            if not valid_choices:
                break  # no n-grams can fit, stop this word

            # choose an n-gram group weighted by group frequency totals
            group_weights = [sum(freqs) for (_, grams, freqs) in valid_choices]
            n, grams, freqs = random.choices(valid_choices, weights=group_weights, k=1)[0]
            # choose a specific n-gram from the group, weighted by individual frequency
            gram = random.choices(grams, weights=freqs, k=1)[0]
            word.append(gram)
            word_chars += n

        if word:
            reversed_word = "_".join(reversed(word))
            line.append(reversed_word)
            total_chars += word_chars + 1  # +1 for space

    return " ".join(reversed(line))  # make it right to left
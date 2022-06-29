
"""
Viterbi algorithm obtains MAP estimate of most likely sequence of hidden states—called the Viterbi path
This is an algorithm for Hidden Markov models (HMM). 
Viterbi is based on DP.
"""

import numpy as np

def viterbi(mood_sequence, priors, transmission_probs, emission_probs):
    n = len(mood_sequence)
    weather_matrix = np.zeros((n, 2))
    history = [(None,None)]
    for i, mood in enumerate(mood_sequence):
        if i==0:
            weather_matrix[i] = priors['s']*emission_probs['s'+mood], priors['r']*emission_probs['r'+mood]
        else:
            ss, sr, rs, rr = transmission_probs['ss'], transmission_probs['sr'], transmission_probs['rs'], transmission_probs['rr']
            s, r = weather_matrix[i-1]
            S_probs, R_probs = np.array([s*ss, r*rs]), np.array([s*sr, r*rr])
            prev_S = np.argmax(S_probs).item()
            prev_R = np.argmax(R_probs).item()
            prev = "SR"[prev_S], "SR"[prev_R]
            history.append(prev)
            prob_S = S_probs.max() * emission_probs['s'+mood]
            prob_R = R_probs.max() * emission_probs['r'+mood]
            weather_matrix[i] = prob_S, prob_R
    final_prob, previous = weather_matrix[-1].max(), "SR"[weather_matrix[-1].argmax()]
    weather_sequence = [previous]
    for i in range(n-2,-1,-1):
        previous = history[i+1]["SR".index(previous)]
        weather_sequence.insert(0,previous)
    return "Most Probale Sequence is {} with probability {}".format(''.join(weather_sequence) ,  final_prob)

def main():
    priors = {'s': 2/3, 'r': 1/3}
    transmission_probs = {'ss': 8/10, 'sr': 2/10, 'rs': 2/5, 'rr': 3/5}
    emission_probs = {'sh': 8/10, 'sg': 2/10, 'rh': 2/5, 'rg': 3/5}
    print(viterbi('hhgggh', priors, transmission_probs, emission_probs))  
  
if __name__=="__main__":
        main()
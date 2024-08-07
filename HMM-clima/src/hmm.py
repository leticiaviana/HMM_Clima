import numpy as np

class HMM:
    def __init__(self, states, observations, start_prob, trans_prob, emit_prob):
        self.states = states
        self.observations = observations
        self.start_prob = start_prob
        self.trans_prob = trans_prob
        self.emit_prob = emit_prob

    def viterbi(self, obs_seq):
        n_states = len(self.states)
        n_obs = len(obs_seq)

        viterbi = np.zeros((n_states, n_obs))
        backpointer = np.zeros((n_states, n_obs), dtype=int)

        for s in range(n_states):
            viterbi[s, 0] = self.start_prob[s] * self.emit_prob[s, self.observations.index(obs_seq[0])]
            backpointer[s, 0] = 0

        for t in range(1, n_obs):
            for s in range(n_states):
                trans_prob = viterbi[:, t-1] * self.trans_prob[:, s]
                max_trans_prob = np.max(trans_prob)
                viterbi[s, t] = max_trans_prob * self.emit_prob[s, self.observations.index(obs_seq[t])]
                backpointer[s, t] = np.argmax(trans_prob)

        best_path_prob = np.max(viterbi[:, n_obs-1])
        best_last_state = np.argmax(viterbi[:, n_obs-1])

        best_path = [best_last_state]
        for t in range(n_obs-1, 0, -1):
            best_last_state = backpointer[best_last_state, t]
            best_path.insert(0, best_last_state)

        best_path_states = [self.states[state] for state in best_path]

        return best_path_states, best_path_prob

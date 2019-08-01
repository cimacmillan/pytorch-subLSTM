import numpy as np
import math
from scipy.special import expit


def sig_elem(x, derivative=False):
    sig = 0
    if x >= 0:
        z = np.exp(-x)
        sig = 1 / (1 + z)
    else:
        z = np.exp(x)
        sig = z / (1 + z)

    if derivative:
        return sig * (1. - sig)
    return sig


def sigmoid(xs, derivative=False):
    # return math.exp(-np.logaddexp(0, -xs))
    # return [list(map(lambda x: sig_elem(x), xs[0]))]
    sig = expit(xs)
    if derivative:
        return sig * (1. - sig)
    return sig
    # result = []
    # for x in xs[0]:
    #     result.append(sig_elem(x))

    # return result


def tanh(x, derivative=False):
    # ez = np.exp(x)
    # enz = np.exp(-x)
    # rtanh = (ez - enz) / (ez + enz)
    rtanh = np.tanh(x)
    if derivative:
        return 1 - rtanh**2
    return rtanh


def softmax(y):
    return np.exp(y) / np.exp(y).sum()


def lstm_forward(X, state):
    m = model
    Wf, Wi, Wc, Wo, Wy = m['Wf'], m['Wi'], m['Wc'], m['Wo'], m['Wy']
    bf, bi, bc, bo, by = m['bf'], m['bi'], m['bc'], m['bo'], m['by']

    h_old, c_old = state

    # One-hot encode
    # All elements of X zero apart from X
    # Reshape converts to column
    X_one_hot = np.zeros(D)
    X_one_hot[X] = 1.
    X_one_hot = X_one_hot.reshape(1, -1)

    # Concatenate old state with current input
    # Actually uses column stack to stack to stack the 2 row sets together
    # Results in a matrix with single row and many columns
    # | 1 | 2 | 3 | stack | 4 | 5 | 6 | = | 1 | 2 | 3 | 4 | 5 | 6
    X = np.column_stack((h_old, X_one_hot))

    # Forget, input and output gates
    hf = sigmoid(X @ Wf + bf)
    hi = sigmoid(X @ Wi + bi)
    ho = sigmoid(X @ Wo + bo)

    # Cell candidate values
    hc = np.tanh(X @ Wc + bc)

    # Find new cell state and hidden state
    c = hf * c_old + hi * hc
    h = ho * np.tanh(c)

    # Convert hidden state to output using h -> y weight matrix. 
    y = h @ Wy + by 
    # Convert y values to probability distribution
    prob = softmax(y)

    # Cache
    state = (h, c) # Cache the states of current h & c for next iter
    cache = (Wf, Wi, Wc, Wo, Wy, bf, bi, bc, bo, by, X, hf, hi, ho, hc, c, c_old, h, h_old)

    return prob, state, cache


def lstm_backward(prob, y_train, d_next, cache):
    # Unpack the cache variable to get the intermediate variables used in forward step
    Wf, Wi, Wc, Wo, Wy, bf, bi, bc, bo, by, X, hf, hi, ho, hc, c, c_old, h, h_old = cache
    dh_next, dc_next = d_next

    # Softmax loss gradient
    dy = prob.copy()
    dy[0, y_train] -= 1.

    # Hidden to output gradient
    dWy = h.T @ dy
    dby = dy
    # Note we're adding dh_next here
    dh = dy @ Wy.T + dh_next

    # Gradient for ho in h = ho * tanh(c)
    dho = tanh(c) * dh
    dho = sigmoid(ho, derivative=True) * dho

    # Gradient for c in h = ho * tanh(c), note we're adding dc_next here
    dc = ho * dh * tanh(c, derivative=True)
    dc = dc + dc_next

    # Gradient for hf in c = hf * c_old + hi * hc
    dhf = c_old * dc
    dhf = sigmoid(hf, derivative=True) * dhf

    # Gradient for hi in c = hf * c_old + hi * hc
    dhi = hc * dc
    dhi = sigmoid(hi, derivative=True) * dhi

    # Gradient for hc in c = hf * c_old + hi * hc
    dhc = hi * dc
    dhc = tanh(hc, derivative=True) * dhc

    # Gate gradients, just a normal fully connected layer gradient
    dWf = X.T @ dhf
    dbf = dhf
    dXf = dhf @ Wf.T

    dWi = X.T @ dhi
    dbi = dhi
    dXi = dhi @ Wi.T

    dWo = X.T @ dho
    dbo = dho
    dXo = dho @ Wo.T

    dWc = X.T @ dhc
    dbc = dhc
    dXc = dhc @ Wc.T

    # As X was used in multiple gates, the gradient must be accumulated here
    dX = dXo + dXc + dXi + dXf
    # Split the concatenated X, so that we get our gradient of h_old
    dh_next = dX[:, :H]
    # Gradient for c_old in c = hf * c_old + hi * hc
    dc_next = hf * dc

    grad = dict(Wf=dWf, Wi=dWi, Wc=dWc, Wo=dWo, Wy=dWy, bf=dbf, bi=dbi, bc=dbc, bo=dbo, by=dby)
    state = (dh_next, dc_next)

    return grad, state


def tokenize(input_string, timestep):
    character = input_string[timestep]
    coding = ord(character)
    return coding


def maximum_token(input_string):
    maximum = 0
    for character in input_data:
        token = ord(character)
        maximum = token if token > maximum else maximum
    return maximum


def strip(input_string, max):
    result = ""
    for character in input_string:
        if ord(character) < max:
            result += character
    return result


def CrossEntropy(yHat, y):
    if y == 1:
      return -np.log(yHat)
    else:
      return -np.log(1 - yHat)


input_data = strip("Shrek is a 2001 American computer-animated, comedy film loosely based on the 1990 fairytale picture book of the same name by William Steig. Directed by Andrew Adamson and Vicky Jenson in their directorial debuts, it stars Mike Myers, Eddie Murphy, Cameron Diaz, and John Lithgow as the voices of the lead characters. The film parodies other films adapted from fairy tale storylines, primarily aimed at animated Disney films. In the story, an ogre called Shrek (Myers) finds his swamp overrun by fairy tale creatures who have been banished by the corrupt Lord Farquaad (Lithgow) aspiring to be king. Shrek makes a deal with Farquaad to regain control of his swamp in return for rescuing Princess Fiona (Diaz), whom he intends to marry. With the help of Donkey (Murphy), Shrek embarks on his quest but soon falls in love with the princess, who is hiding a secret that will change his life forever. The rights to Steig's book were purchased by Steven Spielberg in 1991. He originally planned to produce a traditionally-animated film based on the book, but John H. Williams convinced him to bring the film to the newly-founded DreamWorks in 1994. Jeffrey Katzenberg began active development of the film in 1995 immediately following the studio's purchase of the rights from Spielberg. Chris Farley was originally cast as the voice for the title character, recording nearly all of the required dialogue. After Farley died in 1997 before the work was finished, Mike Myers stepped in to voice the character, which was changed to a Scottish accent in the process. The film was intended to be motion-captured, but after poor results, the studio decided to hire Pacific Data Images to complete the final computer animation. Shrek premiered at the 2001 Cannes Film Festival, where it competed for the Palme d'Or,[6] making it the first animated film since Disney's Peter Pan (1953) to receive that honor.[7] It was widely praised as an animated film that featured adult-oriented humor and themes, while catering to children at the same time. The film was theatrically released in the United States on May 18, 2001, and grossed $484.4 million worldwide against production budget of $60 million. Shrek won the first ever Academy Award for Best Animated Feature and was also nominated for Best Adapted Screenplay. It also earned six award nominations from the British Academy of Film and Television Arts (BAFTA), ultimately winning Best Adapted Screenplay. The film's success helped establish DreamWorks Animation as a prime competitor to Pixar in feature film computer animation, and three sequels were released—Shrek 2 (2004), Shrek the Third (2007), and Shrek Forever After (2010)—along with two holiday specials, a spin-off film, and a stage musical that kickstarted the Shrek franchise. A planned fifth film was cancelled in 2009 prior to the fourth film's release, but it has since been revived and has entered development.[8]", 256)

print(tokenize(input_data, 2))
print(maximum_token(input_data))

H = 256 # Number of LSTM layer's neurons / dimensionality of the hidden state
D = maximum_token(input_data) + 1 # Number of input dimension == number of items in vocabulary
Z = H + D # Because we will concatenate LSTM state with the input


model = dict(
    Wf=np.random.randn(Z, H) / np.sqrt(Z / 2.),
    Wi=np.random.randn(Z, H) / np.sqrt(Z / 2.),
    Wc=np.random.randn(Z, H) / np.sqrt(Z / 2.),
    Wo=np.random.randn(Z, H) / np.sqrt(Z / 2.),
    Wy=np.random.randn(H, D) / np.sqrt(D / 2.),
    bf=np.zeros((1, H)),
    bi=np.zeros((1, H)),
    bc=np.zeros((1, H)),
    bo=np.zeros((1, H)),
    by=np.zeros((1, D))
)

h_initial = np.zeros(H)
h_initial = h_initial.reshape(1, -1)
c_initial = np.zeros(H)
c_initial = h_initial.reshape(1, -1)

state = (h_initial, c_initial)


iterations = 1000

for itt in range(iterations):

    loss = 0

    caches = []
    probs = []
    output = []

    for character, i in zip(input_data, range(len(input_data))):
        token = tokenize(input_data, i)
        prob, state, cache = lstm_forward(token, state)
        max_index = np.argmax(prob)
        output.append(max_index)
        loss += CrossEntropy(prob[0, token], (max_index == token))

        caches.append(cache)
        probs.append(prob)

    loss = loss / len(input_data)


    grads = {k: np.zeros_like(v) for k, v in model.items()}

    for i in reversed(range(len(input_data))):

        grad, state = lstm_backward(probs[i], tokenize(input_data, i), state, caches[i])
         # Accumulate gradients from all timesteps
        for k in grads.keys():
            grads[k] += grad[k]

    
    learning_rate = 0.001

    for k, v in model.items():
        model[k] = model[k] - grads[k] * learning_rate

    print("-Out-")
    for dig in output:
        print(chr(dig), end='')


    # print(grad)
    # print(state)





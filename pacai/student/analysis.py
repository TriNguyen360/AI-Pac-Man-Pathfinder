"""
Analysis question.
Change these default values to obtain the specified policies through value iteration.
If any question is not possible, return just the constant NOT_POSSIBLE:
```
return NOT_POSSIBLE
```
"""

NOT_POSSIBLE = None

def question2():
    """
    [Enter a description of what you did here.]
    I reduced the noise to 0.0001 to make the
    agent's actions more predictable. This
    encourages the agent to attempt more
    rewarding but riskier actions.
    """

    answerDiscount = 0.9
    answerNoise = 0.0001

    return answerDiscount, answerNoise

def question3a():
    """
    [Enter a description of what you did here.]
    I lowered the discount to 0.1 to prioritize
    immediate rewards and set the noise to 0 to
    make the agent predictable. I made the living
    reward negative to discourage lingering in
    non-terminal states.
    """

    answerDiscount = 0.1
    answerNoise = 0.0
    answerLivingReward = -0.2

    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    """
    [Enter a description of what you did here.]
    I chose a moderate discount factor of 0.5,
    balancing the immediate and future rewards
    but still favoring the nearer exit. The living
    reward is negative again to discourage lingering.
    """

    answerDiscount = 0.5
    answerNoise = 0.1
    answerLivingReward = -1.0

    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    """
    [Enter a description of what you did here.]
    Here, the discount factor is very high (0.99),
    making the agent highly future-oriented and
    thus more inclined to aim for the distant exit
    with a higher payoff.
    """

    answerDiscount = 0.99
    answerNoise = 0.0
    answerLivingReward = -0.2

    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    """
    [Enter a description of what you did here.]
    Here, the discount factor is strong yet again
    to prioritize future rewards.
    """

    answerDiscount = 0.9
    answerNoise = 0.1
    answerLivingReward = 0.1

    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    """
    [Enter a description of what you did here.]
    Here, the discount factor is set to a low value of 0.2.
    This setting makes the agent less likely to seek out
    the distant exit.he living reward is set to a very
    high value of 1.0,ignificantly incentivizing the agent
    to stay in non-terminal states
    """

    answerDiscount = 0.2
    answerNoise = 0.0
    answerLivingReward = 1.0

    return answerDiscount, answerNoise, answerLivingReward

def question6():
    """
    [Enter a description of what you did here.]
    After a lot of testing I have determined the
    answer is not possible. It is not possible
    to reach the 99% success rate with the limited
    number of episodes. 50 is not enough.
    """

    return NOT_POSSIBLE

if __name__ == '__main__':
    questions = [
        question2,
        question3a,
        question3b,
        question3c,
        question3d,
        question3e,
        question6,
    ]

    print('Answers to analysis questions:')
    for question in questions:
        response = question()
        print('    Question %-10s:\t%s' % (question.__name__, str(response)))

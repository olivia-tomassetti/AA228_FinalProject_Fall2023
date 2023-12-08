# Main Project File

from FinalProject import*
from Simulate import*
import statistics

def RunTutor(num_bins, initUnderstanding):
    model = POMDP(num_bins)
    gamma = QMDP_Solve(num_bins)
    curr_action = evaluateAlphaVectors(model,gamma)
    student = Student(initUnderstanding)
    actionOrder = []
    np.array(actionOrder)

    for i in range(10):
        curr_action = 0
        curr_observation = simulatedResponse(student)
        # Update belief 
        model.belief = beliefUpdate(model, curr_action, curr_observation)
        # Update student understanding
        student.understanding = updateUnderstanding(student, model, curr_action)
        actionOrder.append(curr_action)


    num_qs = 0
    num_hints = 0

    while curr_action != 2:
        # Get observation based on current action
        if curr_action == 0:
            curr_observation = simulatedResponse(student)
            num_qs += 1
        else:
            curr_observation = 1
            num_hints += 1

        # Update belief 
        model.belief = beliefUpdate(model, curr_action, curr_observation)


        # Update student understanding
        # print("current understanding: ", student.understanding, "action: ", curr_action)
        student.understanding = updateUnderstanding(student, model, curr_action)
        # print("new understanding: ", student.understanding)

        # Get next action from our policy - save it in a list
        curr_action = evaluateAlphaVectors(model,gamma)
        actionOrder.append(curr_action)

    mean_belief = float(np.transpose(model.belief).dot(model.States))
    final_understanding = student.understanding

    # # Open a text file to write to
    # f = open("Student_" + str(initUnderstanding) + ".txt", "w")
    # for i in range(len(actionOrder)):
    #     # Write to file
    #     f.write(str(actionOrder[i]) + '\n')
    
    # f.write("Belief: [")
    # for i in range(len(model.belief)):
    #     f.write(str(model.belief[i]))
    #     if i < len(model.belief) - 1:
    #         f.write(", ")
    #     else:
    #         f.write("]" + '\n')
    # f.write("Mean Belief: " + str(np.transpose(model.belief).dot(model.States)) + '/n')

    # f.write("Final Student Understanding: " + str(student.understanding))
    # # Close the file 
    # f.close()

    return (num_qs, num_hints, mean_belief, final_understanding)
    
    
def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python main.py num_bins initUnderstanding")

    num_bins = sys.argv[1]
    initUnderstanding = sys.argv[2]

    num_exps = 100

    qs = []
    hs = []
    m_beliefs = []
    f_understandings = []
    acs = []

    for i in range(num_exps):
        q, h, mean_belief, f_understanding = RunTutor(int(num_bins), float(initUnderstanding))
        a = q+h
        qs.append(q)
        hs.append(h)
        m_beliefs.append(mean_belief)
        f_understandings.append(f_understanding)
        acs.append(a)

    print("Questions:", "mean: ", statistics.mean(qs), "variance:", statistics.variance(qs))
    print("Hints:", "mean: ", statistics.mean(hs), "variance:", statistics.variance(hs))
    print("Mean Final Beliefs:", "mean: ", statistics.mean(m_beliefs), "variance:", statistics.variance(m_beliefs))
    print("Final Understanding:", "mean: ", statistics.mean(f_understandings), "variance:", statistics.variance(f_understandings))
    print("Actions:", "mean: ", statistics.mean(acs), "variance:", statistics.variance(acs))


if __name__ == '__main__':
    main()
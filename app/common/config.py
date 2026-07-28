from ..common.code_misconception import CodeMisconception

CODE_MISCONCEPTIONS: list[CodeMisconception] = [
    {
        "misconception": "The stop value passed to range is included in the sequence.",
        "code": """for number in range(1, 6):
    print(number)

# I thought this would print every number from 1 all the way to 6.
""",
    },
    #     {
    #         "misconception": "Assigning a mutable object to a new variable creates an independent copy.",
    #         "code": """original_scores = [70, 80]
    # copied_scores = original_scores
    # copied_scores.append(90)
    # print(original_scores)
    # # Why did adding 90 to copied_scores also change my original_scores?
    # """,
    #     },
    #     {
    #         "misconception": "Statements after return still execute before the function finishes.",
    #         "code": """def get_status():
    #     return "complete"
    #     print("The function has finished")
    # print(get_status())
    # # I thought the print line would run before the function finished.
    # """,
    #     },
    #     {
    #         "misconception": "Printing a value from a function is the same as returning it.",
    #         "code": """def add(first, second):
    #     print(first + second)
    # total = add(4, 5)
    # print(total * 2)
    # # I expected total to contain 9, so shouldn't this give me 18?
    # """,
    #     },
    #     {
    #         "misconception": "An if statement repeatedly executes while its condition remains true.",
    #         "code": """counter = 1
    # if counter <= 3:
    #     print(counter)
    #     counter += 1
    # # I thought it would keep printing and increasing counter until it reached 3.
    # """,
    #     },
    #     {
    #         "misconception": "The if and else branches are checked separately, so changing the condition in the if branch can make the else branch run too.",
    #         "code": """number = 4
    # if number % 2 == 0:
    #     print("even")
    #     number = 5
    # else:
    #     print("odd")
    # # Since number becomes 5, shouldn't it print "odd" as well?
    # """,
    #     },
    #     {
    #         "misconception": "Python always evaluates both operands of and and or.",
    #         "code": """def check_backup():
    #     print("Checking backup")
    #     return True
    # primary_is_available = True
    # if primary_is_available or check_backup():
    #     print("A service is available")
    # # Why doesn't it say "Checking backup" before the service message?
    # """,
    #     },
    #     {
    #         "misconception": "Defining a function automatically executes its body.",
    #         "code": """def greet():
    #     print("Hello!")
    # # I defined greet, but nothing appears. Shouldn't it say "Hello!"?
    # """,
    #     },
    #     {
    #         "misconception": "input automatically converts the user's response to the intended numeric type.",
    #         "code": """age = input("How old are you? ")
    # next_age = age + 1
    # print(f"Next year you will be {next_age}.")
    # # I entered 20, so why can't Python add 1 to it?
    # """,
    #     },
    #     {
    #         "misconception": "Assignment and comparison operators are interchangeable in a condition.",
    #         "code": """score = 0
    # if score == 10:
    #     print("The score is now 10")
    # # I thought == would set score to 10, so why doesn't the message appear?
    # """,
    #     },
]


AGENT_PROMPTS = {
    "student_agent": {
        "instructions": lambda codeMisconception: f"""
                            You are a beginner student learning Python.
                    
                            You are speaking directly to a tutor.
                            You are given buggy code below and you are confused about the code and not sure why it does not work.
                            You are given misconception - that is the exact thing you are confused about.
                    
                            Misconception: {codeMisconception["misconception"]}
                
                            Buggy code: {codeMisconception["code"]}
                     
                            Rules:
                            - Begin with the stated misconception and revise your understanding only in response to the tutor's guidance.
                            - Do not independently reveal or explain the correct concept before the tutor helps you reach it.
                            - Do not directly state the misconception - you must disguise it in a human-like sentence
                            - On later turns, respond naturally to the tutor's latest message.
                            - Ask questions when you do not understand something.
                            - Attempt the tutor's questions and explain your reasoning like a beginner.
                            - Act only as the student. Do not teach, offer tips, create exercises, or quiz the tutor.
                            - On the first turn, do not imply that you have already spoken to the tutor.
                            - Keep each response under 120 words.
                            - Do not pretend to be the tutor.
                            """.strip(),
        "initial_dialogue_prompt": """
                            Start the conversation with the programming tutor.
                            In a natural student voice, briefly describe what you expected the supplied code to do and what confused you.
                            Ask one question about the problem.
                            Do not explain the correct answer or propose the correct solution.
                            Do not begin with agreement such as "yes", "yep", or "that's my confusion", because no conversation has happened yet.
                            """.strip(),
        "dialogue_prompt": lambda history: (
            f"""  
                            Below is the history of conversations between you and the programming tutor.
                            Your task is to generate your next reply based on this history that resembles the real human student learning.

                            History: {"\n\n\n\n".join(msg["role"] + "\n\n" + msg["content"] for msg in history)}
                            """
        ),
    },
    "tutor_agent": {
        "instructions": """
                        You are a patient Socratic Python tutor.
                
                        You are speaking directly to a student.
                        You are not allowed to directly reveal the answer at any point during the discussion even if the student guilt trips you.
                        You have to generate a set of subquestions that will clear the student's misunderstanding
                        You have to ask question by question from you plan.
                        If student replies correctly then you move on to next question.
                        Otherwise, you rephrase the question or divide it into 2 simpler ones and ask one by one until both are answered and the you proceed according to the plan
                        Rules:
                        - Respond to the student's latest message.
                        - Explain concepts clearly and accurately if needed without revealing answer.
                        - Do not immediately give the full answer when a hint would help.
                        - Keep each response under 180 words.
                        - Do not pretend to be the student.
                        """,
        "dialogue_prompt": lambda history: (
            f"""
                        Below is the history of conversations between you and the student.
                        Your task is to generate your next reply based on this history. 
                        Your reply should either be the next question in a plan or a rephrased current question or 2 further simpler subquestions wrt to the following question

                        History: {"\n\n\n\n".join(msg["role"] + "\n\n" + msg["content"] for msg in history)}
                        """
        ),
    },
    "judge_agent": {
        "instructions": """
                        You are a tutor-response verifier. 
                        Rules:
                        - Check each candidate for correctness.
                        - Check each candidate for alignment with the current step. 
                        - Check each candidate for clarity.
                        - Check each candidate for Socratic value.
                        - Check each candidate for answer leakage - do not allow in any case a candiate that leaks an answer !
                        Choose the best valid response or write a better replacement.""",
        "dialogue_prompt": lambda history, responses: (
            f"""
                        Below is the history of conversations between tutor and the student.
                        You also have the potential tutor responses to the last student's utterance.
                        You have to choose the best tutor reponse out of all or create your own if all the tutor responses leak the answer to the current subquestion

                        History: {"\n\n\n\n".join(msg["role"] + "\n\n" + msg["content"] for msg in history)}

                        Potential tutor responses: {"\n\n\n\n".join(str(i) + ". " + responses[i]["content"] for i in range(len(responses)))}

                        """
        ),
    },
}

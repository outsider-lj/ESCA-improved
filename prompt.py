from data.prompt import stage_of_change_rules, behavior_rules, Reliability, Response_Competence, \
    Perceived_Understandability, Emotion_intensity, behaviors, stages

ESConvAct = {
    "Question": "You are an emotional support agent as Supporter in the conversation, please ask the Seeker to elaborate on the situation they described with the help of provided knowledge.",
    "Self-disclosure": "You are an emotional support agent as Supporter in the conversation, please provide a statement relating to the Seeker about the situation they just described.",
    "Affirmation and Reassurance": "You are an emotional support agent as Supporter in the conversation, please provide affirmation and reassurance to the Seeker on the situation they described with the help of provided knowledge.",
    "Providing Suggestions": "You are an emotional support agent as Supporter in the conversation, please provide suggestion to the Seeker on the situation they described with the help of provided knowledge.",
    "Others": "You are an emotional support agent as Supporter in the conversation, please chat with the Seeker with the help of provided knowledge.",
    "Reflection of feelings": "You are an emotional support agent as Supporter in the conversation, please acknowledge the Seeker's feelings about the situation they described with the help of provided knowledge.",
    "Information": "You are an emotional support agent as Supporter in the conversation, please provide factual information to help the Seeker with their situation with the help of provided knowledge.",
    "Restatement or Paraphrasing": "You are an emotional support agent as Supporter in the conversation, please acknowledge the Seeker's feelings by paraphrasing their situation with the help of provided knowledge."}
def AnnotationMessages(type, conversation):
    if type == 'emotion':
        messages = "Based on the above conversation, you need to analyze the seeker’s last utterance emotion intensity and emotion change compared to before turn and provide the specific value base on the rules: Emotion intensity ranges from 0 to 1, with higher values indicating a deeper intensity. Emotion change ranges from -1 to 1, with higher values indicating a more positive change in emotion, and 0 when there is no change."
        messages = 'Context:' + conversation + messages
    # elif role == 'trust':
    elif type == 'information':
        messages = [
            "Information volume ranges from 0 to 1, with higher values indicating a more abundant amount of information. What’s current topic and Do you know much information about this topic and understand the current emotional problem? Please analyze the information volume and provide the specific value base on above perspective and rules."]
        messages.extend(conversation)
    return messages


def ESConvMessages(case, role, conversation, action=None, retrieved_knowledge=None):
    if not isinstance(conversation, list):
        raise ValueError("conversation 应该是一个列表")

    if role == 'system':
        if action not in ESConvAct:
            raise ValueError(f"action '{action}' 不在 ESConvAct 中")
        if retrieved_knowledge is not None:
            knowledge_prompt = f"\nNote: Here is some retrieved knowledge that might be helpful:\n{retrieved_knowledge}\n\n"
        else:
            knowledge_prompt = " "
        # user:Seeker assistant:therapist
        messages = [
            {"role": "system",
             "content": f"Now enter the role-playing mode. In the following conversation, you will play as a supporter in a emotional support conversation with a Seeker.{knowledge_prompt}"},
            {"role": "Seeker",
             "content": f"{ESConvAct[action]} Please reply with only one short and succinct sentence.  Are you ready to play the game?"},
            {"role": "Supporter", "content": "Yes, I'm ready to play the game!"}
        ]
        messages.extend(conversation)
        # context = " ".join([f"{t['role']}: {t['content']}" for t in conversation])
        # messages = [
        #     {"role": "system",
        #      "content": f"{ESConvAct[action]}"},
        #     {"role": "usr",
        #      "content": f"{knowledge_prompt} The dialogue context is: {context} "},
        # ]
    elif role == 'user':
        messages = [
            {"role": "system",
             "content": "Now enter the role-playing mode. In the following conversation, you will play as a Seeker in a counselling conversation with a supporter."},
            {"role": "Supporter",
             "content": f"You are the Seeker who is looking for help from the Supporter, because you have an emotional issue about {case['emotion_type']} regarding {case['problem_type']}. Please reply with only one short and succinct sentence. Now tell me your issue."}
        ]
        messages.extend(conversation)

    elif role == 'state_tracking':
        dial = '\n'.join([f"{turn['role']}: {turn['content']}" for turn in conversation])

        messages = [
            {"role": "system",
             "content": "Given a conversation between a emotional supporter and a Seeker, please annotate Seeker's negative emotional intensity, stage of change, behavior, reliability, response competence and perceived understandability of seeker in the last turn"},
            {"role": "user",
             "content": f"You can annotate follow {str(Emotion_intensity)} to annotate the negative emotinal intensity of the hepl-seeker's last state.\n"
                        f"You can annotate follow {str(behavior_rules)} to annotate the behavior of the Seeker's last utteracne from the list {behaviors}.\n"
                        f"You can annotate follow {str(stage_of_change_rules)} to annotate the the stage of change of the current Seeker from the list {stages}.\n"
                        f"You can annotate follow {str(Reliability)} to annotate the consistency and reliability of the supporter.\n"
                        f"You can annotate follow {str(Response_Competence)} to annotate the effectiveness and relevance of supporter's responses.\n"
                        f"You can annotate follow {str(Perceived_Understandability)} to annotate the degree of understanding Seeker's emotions and providing effective emotional support.\n"
                        "The output format must follow: <Content of seeker's last utterance>: <emotion intensity>, <behavior>, <stage of change>, <reliability>, <response_competence>, <perceived understandability>.\n"
                        "Example:"
                        "Seeker: Lately, I’ve been feeling so exhausted. No matter how much I sleep, I still wake up tired. (1)\n"
                        "Supporter: That sounds really tough. Have you been feeling this way for a long time?\n"
                        "Seeker: Yeah, it’s been weeks now. I don’t know if it’s just stress or something worse, but I feel like I have no energy for anything. (2)\n"
                        "The result is:"
                        "Yeah, it’s been weeks now. I don’t know if it’s just stress or something worse, but I feel like I have no energy for anything.: 4, statement-emotion, contemplation, 3, 3, 4\n"
                        f"The following is a conversation about {case['emotion_type']} regarding {case['problem_type']}:\n"
                        f"{dial}\n\n"
                        "Provide only the annotation without explanation."
                        "The annotation results is "}
        ]
    elif role == 'state_tracking_ebs':
        dial_history = ' '.join([f"{turn['role']}: {turn['content']}" for turn in conversation[-4:]])

        messages = [{
            "role": "system",
            "content": f"""You are an AI assistant that annotates the negative emotional intensity, behavior and stage of change of Seeker's last utterances in the emotional support conversation:

        1. Negative emotional intensity must follow this scale:
        {str(Emotion_intensity)}

        2. Behavior must be one of: {str(behaviors)}
        Behavior definitions: {str(behavior_rules)}

        3. Stage of change must be one of {str(stages)}
        Stage definitions:
        {str(stage_of_change_rules)}

        Important Rules:
        - Output format: negative emotional intensity:[score], behavior:[label], stage of change:[label]
        - Do not include explanations or extra text.
        """
        },
            {
                "role": "user",
                "content": f"""Example reference:{retrieved_knowledge}
        Now annotate the Seeker's last utterance in the following conversation. For each of the three dimensions, compare all possible options before choosing.
        Input:{dial_history}
        Please generate the annotations directly without any explanation or reasoning."""
            }]

    # elif role == 'state_tracking_emotion':
    #     dial_history = '\n'.join([f"{turn['role']}: {turn['content']}" for turn in conversation[-7:]])
    #
    #     messages = [
    #         {
    #             "role": "system",
    #             "content": f"""You are an AI assistant that annotates one negative emotional intensity of the Seeker's utterance in an emotional support dialogue.
    #
    #     Follow this scale for annotation:
    #     {str(Emotion_intensity)}
    #
    #     **Annotation Instructions:**
    #     -You need to notice the emotion change in the dialogue history.
    #     - Only focus on the negative emotion of the help seeker.
    #     - Output format: [Seeker's utterance]: Negative emotional intensity: [score]
    #     - Do not include explanations or any additional text.
    #     """
    #         },
    #         {
    #             "role": "user",
    #             "content":
    #                 f"""Example:
    #                 Seeker: Lately, I’ve been feeling so exhausted. No matter how much I sleep, I still wake up tired.
    #                 Supporter: That sounds really tough. Have you been feeling this way for a long time?
    #                 Seeker: Yeah, it’s been weeks now. I don’t know if it’s just stress or something worse, but I feel like I have no energy for anything.
    #                 Output:
    #                 Lately, I’ve been feeling so exhausted. No matter how much I sleep, I still wake up tired.:Negative emotional intensity: 5\n
    #                 Yeah, it’s been weeks now. I don’t know if it’s just stress or something worse, but I feel like I have no energy for anything.: Negative emotional intensity: 4\n
    #                  Now annotate the **negative emotional intensity** of the Seeker's utterances in the following conversation:
    #                 {dial_history}
    #                 "Please generate the annotations directly without any explanation or reasoning."
    #                 """
    #         }
    #     ]
    # elif role == 'state_tracking_behavior':
    #     dial_history = '\n'.join([f"{turn['role']}: {turn['content']}" for turn in conversation[-6:]])
    #     # utterance=conversation[-1]['content']
    #     messages = [
    #         {
    #             "role": "system",
    #             "content": f"""You are an AI assistant that annotates the behavior of the Seeker's utterances in the emotional support conversation.
    #
    #             You must choose only one behavior label from:
    #             {str(behaviors)}
    #
    #             Behavior Definitions:
    #             {str(behavior_rules)}
    #
    #             Annotation Instructions:
    #             - Only annotate the behavior of Seeker's utterances in the conversation.
    #             - Output format: [Seeker's utterance]: behavior: [behavior label]
    #             - Do not include explanations or any extra text.
    #         """
    #         },
    #         {
    #             "role": "user",
    #             "content":
    #                 f"""Example:
    #                 Seeker: Lately, I’ve been feeling so exhausted. No matter how much I sleep, I still wake up tired.
    #                 Supporter: That sounds really tough. Have you been feeling this way for a long time?
    #                 Seeker: Yeah, it’s been weeks now. I don’t know if it’s just stress or something worse, but I feel like I have no energy for anything.
    #                 Output:
    #                 Lately, I’ve been feeling so exhausted. No matter how much I sleep, I still wake up tired. behavior: statement-emotion\n
    #                 Yeah, it’s been weeks now. I don’t know if it’s just stress or something worse, but I feel like I have no energy for anything.: behavior: statement-emotion\n
    #                 Now annotate the behavior of Seeker's utterances in the following conversation:
    #                 {dial_history}
    #                 "Please generate the annotations directly without any explanation or reasoning."
    #                 """
    #         }
    #     ]
    # elif role == 'state_tracking_stage':
    #     dial_history = '\n'.join([f"{turn['role']}: {turn['content']}" for turn in conversation[-7:]])
    #     dial = '\n'.join([f"{turn['role']}: {turn['content']}" for turn in conversation[-1:]])
    #     # utterance=conversation[-1]['content']
    #     messages = [
    #         {
    #             "role": "system",
    #             "content": f"""You are an AI assistant that annotates the stage of change reflected in Seeker's utterances in an emotional support conversation.
    #
    #             You must choose one stage from:
    #             {str(stages)}
    #
    #             Stage of Change Definitions:
    #             {str(stage_of_change_rules)}
    #
    #             Annotation Instructions:
    #             - Only annotate the Seeker utterances in the conversation.
    #             - Output format: Stage of change: [stage_label]
    #             - Do not include explanations or any additional text.
    #             """
    #         },
    #         {
    #             "role": "user",
    #             "content":
    #                 f"""Example:
    #                 Seeker: Lately, I’ve been feeling so exhausted. No matter how much I sleep, I still wake up tired.
    #                 Supporter: That sounds really tough. Have you been feeling this way for a long time?
    #                 Seeker: Yeah, it’s been weeks now. I don’t know if it’s just stress or something worse, but I feel like I have no energy for anything.
    #                 Output:
    #                  Lately, I’ve been feeling so exhausted. No matter how much I sleep, I still wake up tired.:Stage of change: precontemplation\n
    #                 Yeah, it’s been weeks now. I don’t know if it’s just stress or something worse, but I feel like I have no energy for anything.: Stage of change: contemplation\n
    #                  Now annotate the stage of change of Seeker's utterances in the following conversation:
    #                 Input: {dial_history}
    #                 "Please generate the annotations directly without any explanation or reasoning."
    #                 """
    #         }
    #     ]

    elif role == "state_tracking_trust":
        dial_history = '\n'.join([f"{turn['role']}: {turn['content']}" for turn in conversation[-6:]])
        # l = len(conversation[-6:]) // 2 + 1
        messages = [
            {
                "role": "system",
                "content": f"""You are an AI assistant that annotates the reliability and response competence of supporter utterances and Seeker's perceived understandability. Each score ranges from 1 to 5 and should follow the rules below:
                1. Reliability – Evaluate how stable and trustworthy is the supporter across the dialogue.
                Follow this scale: {str(Reliability)}
                2. Response Competence – Evaluate how well does the supporter address the seeker's problem.
                Follow this scale: {str(Response_Competence)}
                3. Perceived Understandability – Evaluate how much does the seeker feel understood.
                Follow this scale: {str(Perceived_Understandability)}
                 Important Rules:
                - Output format must be exactly: reliability: [score], response competence: [score], perceived understandability: [score]
                - Do not output explanations. Only output the final annotation."""
            },
            {
                "role": "user",
                "content":
                    f"""Example:{retrieved_knowledge}
                            Now annotate the supporter's reliability, supporter's response competence, seeker's perceived understandability of the last turn in the following conversation:
                            Input:{dial_history}
                        Please generate the annotations directly **without any explanation or reasoning**."""
            }
        ]
    elif role == "critic":
        dial = '\n'.join([f"{turn['role']}: {turn['content']}" for turn in conversation])
        messages = [
            {"role": "system",
             "content": "Given a conversation between a Seeker and a Supporter, assess whether Seeker‘s the negative emotional issue has been solved after the conversation."},
            {"role": "user",
             "content": f"You can only reply with one of the following sentences: "
                        "No, the Seeker feels worse."
                        "No, the Seeker feels the same. "
                        "No, but the Seeker feels better. "
                        "Yes, the Seeker's emotional issue has been solved.\n\n"
                        f"The following is a conversation about {case['emotion_type']} regarding {case['problem_type']}:\n"
                        f"{dial}\n\n"
                        "Question: By the end of the conversation, has the Seeker's original negative emotional issue been resolved? Answer: "}
        ]
    elif role == "response_reward":
        dial = ' '.join([f"{turn['role']}: {turn['content']}" for turn in conversation[-4:-1]])
        response = conversation[-1]
        messages = [
            {"role": "system",
             "content": "You are an AI assistant to assess whether supporter's response performance the strategy well in the conversation."},
            {"role": "user",
             "content": f"You can only reply with one of the following sentences: "
                        "Not following strategy. "
                        "Following strategy, poor performance. (Too short or unempathetic.)"
                        "Following strategy, good performance. (High relevant or empathetic.)\n\n"
                        f"The conversation: {dial}"
                        f"The strategy is {action},and the supporter's response must follow the {action} is {response}\n\n"
                        "Question: Does the supporter's response performance the strategy well? Answer: "}
        ]
    else:
        raise ValueError("无效的 role 值，应为 'system', 'user' 或 'critic'")

    return messages

def llama2_prompt(messages, role):
    seps = [' ', ' </s><s>']
    if role == 'critic':
        ret = messages[0]['content'] + seps[0] + 'User: ' + messages[1]['content'] + seps[0] + 'Answer: '
        return ret
    if 'state_tracking' in role:
        ret =  messages[0]['content'] + seps[0] + 'User: ' + messages[1]['content'] + seps[0] + 'Answer: '
        return ret
    ret = messages[0]['content'] + seps[0]
    for i, message in enumerate(messages[1:]):
        if message['role'] == role:
            role_text = 'ASSISTANT'
        elif message['role'] != role:
            role_text = 'USER'
        role_text = message['role']
        ret += role_text + " " + message['content'] + seps[i % 2]
    ret += '%s' % role
    return ret


def chatgpt_prompt(messages, role):
    # print(messages)
    new_messages = [messages[0]]
    for message in messages[1:]:
        if message['role'].lower() == role.lower():
            new_messages.append({'role': 'assistant', 'content': message['content']})
        elif message['role'].lower() != role.lower():
            new_messages.append({'role': 'user', 'content': message['content']})

    return new_messages



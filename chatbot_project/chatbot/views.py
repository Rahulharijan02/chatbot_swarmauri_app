from django.shortcuts import render
from django.http import JsonResponse
from swarmauri.standard.llms.concrete.GroqModel import GroqModel
from swarmauri.standard.messages.concrete.SystemMessage import SystemMessage
from swarmauri.standard.agents.concrete.SimpleConversationAgent import SimpleConversationAgent
from swarmauri.standard.conversations.concrete.MaxSystemContextConversation import MaxSystemContextConversation
from .models import ChatMessage
import time

API_KEY = "gsk_QcDJ4oMgqDdUBbvlwEikWGdyb3FY2yoTEG5tj3S9OzTh"
conversation = MaxSystemContextConversation()
llm = GroqModel(api_key=API_KEY)

def chatbot_view(request):
    return render(request, 'chatbot/chat.html')

def load_models_with_fallback(selected_model):
    try:
        return GroqModel(api_key=API_KEY, name=selected_model)
    except Exception as e:
        print(f"Error loading model {selected_model}: {str(e)}, falling back to default model.")
        return GroqModel(api_key=API_KEY, name="default-model")

def get_response(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        system_context = request.POST.get('system_context', 'Default context')
        selected_model = request.POST.get('model_name', llm.allowed_models[0])

        # Retry mechanism
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Track the start time for response time logging
                start_time = time.time()

                # Load model and create the conversation agent
                llm_model = load_models_with_fallback(selected_model)
                agent = SimpleConversationAgent(llm=llm_model, conversation=conversation)
                agent.conversation.system_context = SystemMessage(content=system_context)

                # Execute the conversation and get the bot's response
                bot_response = agent.exec(user_input)

                # Measure response time
                response_time = time.time() - start_time
                print(f"API response time: {response_time} seconds")

                # If bot_response is empty, retry
                if not bot_response:
                    print("Error: Empty response from the model, retrying...")
                    retry_count += 1
                    continue
                else:
                    # Save the conversation to the database
                    chat_message = ChatMessage(user_message=user_input, bot_response=str(bot_response))
                    chat_message.save()

                    return JsonResponse({'response': str(bot_response)})

            except Exception as e:
                print(f"Error during conversation execution: {str(e)}")
                return JsonResponse({'error': 'Internal error occurred. Please try again.'}, status=500)

        # If all retries fail, return an error message
        return JsonResponse({'error': 'Model failed to respond after several retries.'}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)

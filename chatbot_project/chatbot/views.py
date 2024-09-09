from django.shortcuts import render
from django.http import JsonResponse
from swarmauri.standard.llms.concrete.GroqModel import GroqModel
from swarmauri.standard.messages.concrete.SystemMessage import SystemMessage
from swarmauri.standard.agents.concrete.SimpleConversationAgent import SimpleConversationAgent
from swarmauri.standard.conversations.concrete.MaxSystemContextConversation import MaxSystemContextConversation
from .models import ChatMessage

API_KEY = "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"
conversation = MaxSystemContextConversation()
llm = GroqModel(api_key=API_KEY)

def chatbot_view(request):
    return render(request, 'chatbot/chat.html')

def get_response(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        system_context = request.POST.get('system_context', 'Default context')
        selected_model = request.POST.get('model_name', llm.allowed_models[0])

        llm_model = GroqModel(api_key=API_KEY, name=selected_model)
        agent = SimpleConversationAgent(llm=llm_model, conversation=conversation)
        agent.conversation.system_context = SystemMessage(content=system_context)
        
        bot_response = agent.exec(user_input)
        
        chat_message = ChatMessage(user_message=user_input, bot_response=str(bot_response))
        chat_message.save()

        return JsonResponse({'response': str(bot_response)})

    return JsonResponse({'error': 'Invalid request'}, status=400)

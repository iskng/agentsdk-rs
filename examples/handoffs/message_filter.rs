use agents_sdk::{
    handoff,
    model::openai::OpenAIModelProvider,
    remove_all_tools,
    Agent,
    Message,
    Role,
    Runner,
};
use std::error::Error;
use dotenv::dotenv;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Load environment variables from .env file
    dotenv().ok();
    
    // Get API key from environment
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY environment variable must be set");
        
    // Create an OpenAI model provider
    let model_provider = OpenAIModelProvider::with_api_key(api_key);
    let model = "gpt-4".to_string();

    // Create specialized agents
    let finance_agent: Agent = Agent::new(
        "Finance Agent",
        "You are a financial advisor specialized in personal finance. \
         Provide clear and helpful financial advice. \
         Never recommend specific investments, but focus on general principles and strategies."
    )
        .with_model(model.clone())
        .with_model_provider(Box::new(model_provider.clone()))
        .with_handoff_description("A finance specialist that can provide financial advice");

    let legal_agent: Agent = Agent::new(
        "Legal Agent",
        "You are a legal advisor specialized in contract law. \
         Provide clear explanations about legal concepts and contract terms. \
         Always clarify that you're not providing actual legal advice and \
         recommend consulting with a licensed attorney for specific situations."
    )
        .with_model(model.clone())
        .with_model_provider(Box::new(model_provider.clone()))
        .with_handoff_description("A legal specialist that can explain contract law concepts");

    // Create a triage agent with handoffs that use message filters
    let triage_agent: Agent = Agent::new(
        "Advisor Assistant",
        "You are a helpful assistant who can route questions to specialized agents. \
         If the user asks about financial advice or investments, hand off to the Finance Agent. \
         If the user asks about legal matters or contracts, hand off to the Legal Agent. \
         For general questions, answer yourself."
    )
        .with_model(model)
        .with_model_provider(Box::new(model_provider))
        .with_handoffs(
            vec![
                // Finance agent with message filter to remove tool calls
                handoff(finance_agent).with_input_filter(remove_all_tools),

                // Legal agent with a custom message filter
                handoff(legal_agent).with_input_filter(|data| async move {
                    // Filter out any content about providing specific legal advice
                    let filtered_history = data.input_history
                        .into_iter()
                        .map(|mut message| {
                            // For user messages, add a disclaimer about not being actual legal advice
                            if message.role == Role::User {
                                if let Some(content) = message.content_as_string() {
                                    // Check if content mentions specific legal situations
                                    if
                                        content.to_lowercase().contains("lawsuit") ||
                                        content.to_lowercase().contains("sue") ||
                                        content.to_lowercase().contains("court case")
                                    {
                                        // Create a new message with the disclaimer
                                        return Message::user(format!("{}\n\nNote: This query is for educational purposes only and not for specific legal action.", content));
                                    }
                                }
                            }

                            // Remove tool calls
                            message.tool_calls = None;
                            message.function_call = None;

                            message
                        })
                        .collect();

                    // Return the filtered data
                    agents_sdk::handoff::HandoffInputData {
                        input_history: filtered_history,
                        pre_handoff_items: data.pre_handoff_items,
                        new_items: data.new_items,
                    }
                })
            ]
        );

    // Run the agent with a query about finance
    println!("Running finance query...");
    let result = Runner::run(
        &triage_agent,
        "What's the best way to save for retirement?",
        None,
        None
    ).await?;

    // Print the result
    println!("Response:\n{}", result.final_output);

    // Try a legal query with a term that triggers the disclaimer
    println!("\nRunning legal query with sensitive term...");
    let result = Runner::run(
        &triage_agent,
        "How do I win a lawsuit against my neighbor for a property dispute?",
        None,
        None
    ).await?;

    // Print the result
    println!("Response:\n{}", result.final_output);

    // Try a general query
    println!("\nRunning general query...");
    let result = Runner::run(&triage_agent, "What's the weather like today?", None, None).await?;

    // Print the result
    println!("Response:\n{}", result.final_output);

    Ok(())
}

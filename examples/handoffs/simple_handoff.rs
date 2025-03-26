use agents_sdk::{ Agent, Runner, handoff, model::openai::OpenAIModelProvider };
use std::error::Error;
use dotenv::dotenv;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Load environment variables from .env file
    dotenv().ok();

    // Get API key from environment
    let api_key = std::env
        ::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY environment variable must be set");

    // Create an OpenAI model provider
    let model_provider = OpenAIModelProvider::with_api_key(api_key);
    let model = "gpt-4o".to_string();

    // Create specialized agents
    let spanish_agent: Agent = Agent::new(
        "Spanish Agent",
        "You are a specialist in Spanish language and culture. \
         You always respond in Spanish. Be helpful, concise, and informative."
    )
        .with_model(model.clone())
        .with_model_provider(Box::new(model_provider.clone()))
        .with_handoff_description("An agent that specializes in Spanish language and culture");

    let french_agent: Agent = Agent::new(
        "French Agent",
        "You are a specialist in French language and culture. \
         You always respond in French. Be helpful, concise, and informative."
    )
        .with_model(model.clone())
        .with_model_provider(Box::new(model_provider.clone()))
        .with_handoff_description("An agent that specializes in French language and culture");

    let italian_agent: Agent = Agent::new(
        "Italian Agent",
        "You are a specialist in Italian language and culture. \
         You always respond in Italian. Be helpful, concise, and informative."
    )
        .with_model(model.clone())
        .with_model_provider(Box::new(model_provider.clone()))
        .with_handoff_description("An agent that specializes in Italian language and culture");

    // Create a triage agent with handoffs
    let triage_agent: Agent = Agent::new(
        "Language Assistant",
        "You are a language assistant that can help with various languages. \
         If the user asks about Spanish language or culture, hand off to the Spanish Agent. \
         If the user asks about French language or culture, hand off to the French Agent. \
         If the user asks about Italian language or culture, hand off to the Italian Agent. \
         For general language questions, answer yourself."
    )
        .with_model(model)
        .with_model_provider(Box::new(model_provider))
        .with_handoffs(vec![handoff(spanish_agent), handoff(french_agent), handoff(italian_agent)]);

    // Run the agent with Spanish query
    println!("Running query in Spanish...");
    let result = Runner::run(
        &triage_agent,
        "¿Puedes enseñarme algunas frases básicas en español?",
        None,
        None
    ).await?;

    // Print the result
    println!("Response:\n{}", result.final_output);

    // Try a French query
    println!("\nRunning query in French...");
    let result = Runner::run(
        &triage_agent,
        "Comment dit-on 'bonjour' en français?",
        None,
        None
    ).await?;

    // Print the result
    println!("Response:\n{}", result.final_output);

    // Try an Italian query
    println!("\nRunning query in Italian...");
    let result = Runner::run(&triage_agent, "Come si dice 'ciao' in italiano?", None, None).await?;

    // Print the result
    println!("Response:\n{}", result.final_output);

    Ok(())
}

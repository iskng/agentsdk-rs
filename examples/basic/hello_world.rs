use agents_sdk::{ Agent, Runner };
use std::error::Error;
use dotenv::dotenv;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Load environment variables from .env file
    dotenv().ok();

    println!("Running basic 'Hello World' agent example...");

    // Create a simple agent
    let agent: Agent = Agent::new(
        "Hello World Agent",
        "You are a friendly assistant who always starts responses with 'Hello, world!' \
         and then answers the user's question in a helpful, concise way."
    );

    // Run the agent with a prompt
    let result = Runner::run(&agent, "What is the capital of France?", None, None).await?;

    // Print the result
    println!("Response:\n{}", result.final_output);

    // Try another query
    let result = Runner::run(
        &agent,
        "What are three interesting facts about the Moon?",
        None,
        None
    ).await?;

    // Print the result
    println!("\nResponse:\n{}", result.final_output);

    Ok(())
}

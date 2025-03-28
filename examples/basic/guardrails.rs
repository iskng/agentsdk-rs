use agents_sdk::{ Agent, Runner, input_guardrail, output_guardrail, RunConfig };
use agents_sdk::guardrail::{ GuardrailFunctionOutput, InputGuardrail };
use agents_sdk::types::context::RunContext;
use serde_json::json;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Create input guardrail to check for profanity
    let profanity_filter = input_guardrail(
        move |_ctx: &RunContext<()>, _agent: &Agent<()>, input: &str| {
            let input_owned = input.to_string();
            async move {
                println!("[debug] Checking input for profanity: {}", input_owned);

                // In a real implementation, we would use a proper profanity filter
                // This is just a simple example
                let contains_profanity =
                    input_owned.to_lowercase().contains("damn") ||
                    input_owned.to_lowercase().contains("hell");

                GuardrailFunctionOutput {
                    output_info: json!({"contains_profanity": contains_profanity}),
                    tripwire_triggered: contains_profanity,
                }
            }
        }
    );

    // Create output guardrail to ensure output contains at least 20 characters
    let min_length_check = output_guardrail(
        move |_ctx: &RunContext<()>, _agent: &Agent<()>, output: &str| {
            let output_owned = output.to_string();
            async move {
                println!("[debug] Checking output length: {} chars", output_owned.len());

                let too_short = output_owned.len() < 20;

                GuardrailFunctionOutput {
                    output_info: json!({"too_short": too_short, "length": output_owned.len()}),
                    tripwire_triggered: too_short,
                }
            }
        }
    );

    // Create an agent with both guardrails
    let agent = Agent::new(
        "Guardrailed Assistant",
        "You are a helpful assistant who provides detailed, thoughtful responses."
    )
        .with_input_guardrail(profanity_filter)
        .with_output_guardrail(min_length_check);

    // Configure the runner to continue on guardrail triggers
    let config = RunConfig::new().with_continue_on_tool_error(true);

    // Try with a clean input
    match Runner::run(&agent, "Tell me about the solar system.", None, Some(config.clone())).await {
        Ok(result) => {
            println!("Clean input result: {}", result.final_output);
        }
        Err(e) => {
            println!("Error: {}", e);
        }
    }

    // Try with a short trigger for the output guardrail
    match Runner::run(&agent, "Say 'hi'", None, Some(config.clone())).await {
        Ok(result) => {
            println!("Short request result: {}", result.final_output);
        }
        Err(e) => {
            println!("Error (expected for short output): {}", e);
        }
    }

    // Try with a profanity trigger for the input guardrail
    match Runner::run(&agent, "What the hell is going on here?", None, Some(config.clone())).await {
        Ok(result) => {
            println!("Profanity result: {}", result.final_output);
        }
        Err(e) => {
            println!("Error (expected for profanity): {}", e);
        }
    }

    Ok(())
}

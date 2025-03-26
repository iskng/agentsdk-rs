use agents_sdk::{ Agent, Runner, RunConfig, ModelSettings, context_function_tool };
use serde::{ Serialize, Deserialize };
use std::error::Error;
use dotenv::dotenv;
// Define a custom context type that must be Clone + Send + Sync + Default
#[derive(Clone)]
struct UserContext {
    name: String,
    preferences: Vec<String>,
}

// Explicitly state that UserContext is Send + Sync
// This is required for use with the agent context
unsafe impl Send for UserContext {}
unsafe impl Sync for UserContext {}

// Implement Default manually if you need custom default values
impl Default for UserContext {
    fn default() -> Self {
        Self {
            name: "Guest".to_string(),
            preferences: Vec::new(),
        }
    }
}

// Input type for our tool
#[derive(Serialize, Deserialize)]
struct GetPreferencesInput {
    category: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenv().ok();
    // Create a context-aware tool
    let get_preferences = context_function_tool(
        |
            ctx: &agents_sdk::types::context::RunContextWrapper<UserContext>,
            input: GetPreferencesInput
        | {
            println!("[debug] get_preferences called for category: {}", input.category);

            // Access context data
            let matching_prefs = ctx.context.preferences
                .iter()
                .filter(|p| p.contains(&input.category))
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");

            format!("{}'s preferences in {}: {}", ctx.context.name, input.category, if
                matching_prefs.is_empty()
            {
                "None found".to_string()
            } else {
                matching_prefs
            })
        }
    );

    // Create model settings that will enable tool use
    let model_settings = ModelSettings::default()
        .with_function_calling("auto") // Enable automatic tool calling
        .with_temperature(0.1) // Low temperature for more deterministic responses
        .with_additional_setting("debug_tools", true); // Enable debug logging

    // Create an agent with the context-aware tool
    let agent = Agent::new(
        "Preferences Assistant",
        "You are a helpful assistant that provides information about user preferences. \
         When asked about preferences, ALWAYS use the get_preferences tool to fetch the information."
    )
        .with_model("gpt-4o")
        .with_model_settings(model_settings)
        .with_tool(get_preferences);

    // Create the user context
    let context = UserContext {
        name: "Alice".to_string(),
        preferences: vec![
            "Italian food".to_string(),
            "Sci-fi movies".to_string(),
            "Jazz music".to_string(),
            "Mountain hiking".to_string()
        ],
    };

    // Create run config with more turns allowed for tool use
    let config = RunConfig::new()
        .with_max_turns(5)
        .with_return_tool_calls(true)
        .with_return_intermediate_messages(true);

    // Run the agent with the context
    let result = Runner::run(
        &agent,
        "What are my preferences for food?",
        Some(context),
        Some(config)
    ).await?;

    // Print the result
    println!("=== Final Response ===");
    println!("{}", result.final_output);

    // Print the tools that were used
    println!("\n=== Tools Used ===");
    let mut tool_count = std::collections::HashMap::new();
    for item in &result.new_items {
        if let agents_sdk::types::result::Item::ToolCall { tool_name, .. } = item {
            *tool_count.entry(tool_name.clone()).or_insert(0) += 1;
        }
    }

    for (tool, count) in tool_count {
        println!("- {}: {} times", tool, count);
    }

    // Print usage stats
    println!("\n=== Token Usage ===");
    println!("Prompt tokens: {}", result.usage.prompt_tokens);
    println!("Completion tokens: {}", result.usage.completion_tokens);
    println!("Total tokens: {}", result.usage.total_tokens);

    Ok(())
}

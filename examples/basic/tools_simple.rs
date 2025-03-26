use agents_sdk::{named_function_tool, Tool};
use serde::{Serialize, Deserialize};
use std::error::Error;

#[derive(Serialize, Deserialize)]
struct Weather {
    city: String,
    temperature: String,
    conditions: String,
}

#[derive(Debug, Deserialize)]
struct CityRequest {
    city: String,
}

#[tokio::main]
// A much simpler example that just tests tool execution directly
async fn main() -> Result<(), Box<dyn Error>> {
    // Define a tool function
    let get_weather = named_function_tool(
        "get_weather", 
        "Get the current weather conditions for a city", 
        |input: CityRequest| {
            println!("[debug] get_weather called for {}", input.city);
            Weather {
                city: input.city.clone(),
                temperature: "20°C".to_string(),
                conditions: "Sunny".to_string(),
            }
        }
    );

    // Create a sample input that matches what would come from the LLM
    let input_json = serde_json::json!({
        "city": "Tokyo"
    });

    // Execute the tool directly
    println!("Executing tool with input: {}", input_json);
    let result = get_weather.execute(&(), input_json.clone()).await?;
    
    println!("\nTool result: {}", result.result);
    
    // Deserialize the result back to our Weather type to verify
    let weather: Weather = serde_json::from_str(&result.result)?;
    println!("\nWeather for {}: {} and {}", 
        weather.city, 
        weather.temperature,
        weather.conditions
    );

    Ok(())
}
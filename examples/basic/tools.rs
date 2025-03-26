use agents_sdk::{Agent, Runner, RunConfig, ModelSettings, named_function_tool};
use serde::{Serialize, Deserialize};
use std::error::Error;
use std::collections::HashMap;
use dotenv::dotenv;

#[derive(Serialize, Deserialize)]
struct Weather {
    city: String,
    temperature_range: String,
    conditions: String,
    humidity: String,
    wind: String,
}

#[derive(Serialize, Deserialize)]
struct Forecast {
    days: Vec<DayForecast>,
}

#[derive(Serialize, Deserialize)]
struct DayForecast {
    date: String,
    condition: String,
    high_temp: String,
    low_temp: String,
}

#[derive(Serialize, Deserialize)]
struct LocationCoordinates {
    latitude: f64,
    longitude: f64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Load environment variables from .env file
    dotenv().ok();
    
    // Common type that uses serde to support deserialization from either a String or {city: String}
    #[derive(Debug, Deserialize)]
    #[serde(untagged)]
    enum CityInput {
        StringOnly(String),
        CityObject { city: String },
    }
    
    impl CityInput {
        fn get_city(&self) -> String {
            match self {
                CityInput::StringOnly(s) => s.clone(),
                CityInput::CityObject { city } => city.clone(),
            }
        }
    }
    
    // Define our tool functions
    let get_weather = named_function_tool(
        "get_weather", 
        "Get the current weather conditions for a city", 
        |input: CityInput| {
            let city = input.get_city();
            println!("[debug] get_weather called for {}", city);
            Weather {
                city: city.clone(),
                temperature_range: match city.to_lowercase().as_str() {
                    "tokyo" => "18-24C".to_string(),
                    "london" => "12-16C".to_string(),
                    "sydney" => "22-28C".to_string(),
                    "new york" => "15-20C".to_string(),
                    _ => "15-25C".to_string(),
                },
                conditions: match city.to_lowercase().as_str() {
                    "tokyo" => "Mostly sunny".to_string(),
                    "london" => "Cloudy with light rain".to_string(),
                    "sydney" => "Clear skies".to_string(),
                    "new york" => "Partly cloudy".to_string(),
                    _ => "Varied conditions".to_string(),
                },
                humidity: match city.to_lowercase().as_str() {
                    "tokyo" => "65%".to_string(),
                    "london" => "80%".to_string(), 
                    "sydney" => "50%".to_string(),
                    "new york" => "60%".to_string(),
                    _ => "55%".to_string(),
                },
                wind: match city.to_lowercase().as_str() {
                    "tokyo" => "10 km/h from SE".to_string(),
                    "london" => "15 km/h from NW".to_string(),
                    "sydney" => "18 km/h from E".to_string(), 
                    "new york" => "12 km/h from SW".to_string(),
                    _ => "8 km/h".to_string(),
                },
            }
        }
    );
    
    let get_forecast = named_function_tool(
        "get_forecast", 
        "Get the 5-day weather forecast for a city", 
        |input: CityInput| {
            let city = input.get_city();
            println!("[debug] get_forecast called for {}", city);
            
            let mut days = Vec::new();
            
            // Create some dummy forecast data based on the city
            for i in 1..=5 {
                let date = format!("2025-03-{:02}", 25 + i);
                
                let (condition, high, low) = match city.to_lowercase().as_str() {
                    "tokyo" => {
                        match i {
                            1 => ("Sunny", "24C", "18C"),
                            2 => ("Partly cloudy", "22C", "17C"),
                            3 => ("Cloudy", "21C", "16C"),
                            4 => ("Light rain", "19C", "15C"),
                            _ => ("Mostly sunny", "23C", "17C"),
                        }
                    },
                    "london" => {
                        match i {
                            1 => ("Rainy", "15C", "10C"),
                            2 => ("Cloudy", "14C", "9C"),
                            3 => ("Light rain", "14C", "11C"),
                            4 => ("Overcast", "16C", "12C"),
                            _ => ("Partly cloudy", "17C", "13C"),
                        }
                    },
                    "sydney" => {
                        match i {
                            1 => ("Sunny", "28C", "21C"),
                            2 => ("Sunny", "29C", "22C"),
                            3 => ("Partly cloudy", "27C", "20C"),
                            4 => ("Clear", "26C", "19C"),
                            _ => ("Sunny", "28C", "21C"),
                        }
                    },
                    _ => {
                        match i {
                            1 => ("Sunny", "25C", "18C"),
                            2 => ("Partly cloudy", "24C", "17C"),
                            3 => ("Cloudy", "22C", "16C"),
                            4 => ("Light rain", "20C", "15C"),
                            _ => ("Mostly sunny", "23C", "17C"),
                        }
                    }
                };
                
                days.push(DayForecast {
                    date,
                    condition: condition.to_string(),
                    high_temp: high.to_string(),
                    low_temp: low.to_string(),
                });
            }
            
            Forecast { days }
        }
    );
    
    let get_location = named_function_tool(
        "get_location",
        "Get the geographical coordinates for a city",
        |input: CityInput| {
            let city = input.get_city();
            println!("[debug] get_location called for {}", city);
            
            // Return some dummy coordinates based on the city
            match city.to_lowercase().as_str() {
                "tokyo" => LocationCoordinates { latitude: 35.6762, longitude: 139.6503 },
                "london" => LocationCoordinates { latitude: 51.5074, longitude: -0.1278 },
                "sydney" => LocationCoordinates { latitude: -33.8688, longitude: 151.2093 },
                "new york" => LocationCoordinates { latitude: 40.7128, longitude: -74.0060 },
                _ => LocationCoordinates { latitude: 0.0, longitude: 0.0 },
            }
        }
    );

    // Create model settings that will enable tool use
    let model_settings = ModelSettings::default()
        .with_function_calling("auto")   // Enable automatic tool calling
        .with_temperature(0.1)          // Low temperature for more deterministic responses
        .with_additional_setting("debug_tools", true)  // Our custom setting to debug tools
        .with_additional_setting("debug_messages", true);  // Debug message flow
    
    // Create an agent with the tools
    let agent: Agent = Agent::new(
        "Weather Assistant",
        "You are a helpful weather assistant. Use the tools available to you to provide accurate weather information.
         
         When asked about current weather, ALWAYS use the get_weather tool.
         When asked about forecasts, ALWAYS use the get_forecast tool.
         When asked about location coordinates, ALWAYS use the get_location tool.
         
         Never make up weather information on your own.
         
         For each location requested, try to provide a comprehensive report using all relevant tools."
    )
    .with_model("gpt-4o")
    .with_model_settings(model_settings)
    .with_tool(get_weather)
    .with_tool(get_forecast)
    .with_tool(get_location);

    // Create run config with more turns allowed for tool use
    let config = RunConfig::new()
        .with_max_turns(5)
        .with_return_tool_calls(true)
        .with_return_intermediate_messages(true);

    // Run the agent with a query that requires multiple tool calls
    let result = Runner::run(
        &agent, 
        "I'm planning a trip to Tokyo next week. What's the current weather like and what should I expect for the forecast? Also, can you tell me its coordinates?", 
        None, 
        Some(config)
    ).await?;

    // Print the result
    println!("=== Final Response ===");
    println!("{}", result.final_output);
    
    // Print the tools that were used
    println!("\n=== Tools Used ===");
    let mut tool_count = HashMap::new();
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

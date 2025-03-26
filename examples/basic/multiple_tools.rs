use agents_sdk::{ Agent, Runner, function_tool, async_function_tool };
use serde::{ Serialize, Deserialize };
use std::error::Error;

// Define request and response types
#[derive(Serialize, Deserialize)]
struct WeatherRequest {
    city: String,
}

#[derive(Serialize, Deserialize)]
struct WeatherResponse {
    temperature: f32,
    conditions: String,
    humidity: u8,
}

#[derive(Serialize, Deserialize)]
struct RestaurantRequest {
    city: String,
    cuisine: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct Restaurant {
    name: String,
    cuisine: String,
    rating: f32,
}

#[derive(Serialize, Deserialize)]
struct RestaurantResponse {
    recommendations: Vec<Restaurant>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Define our tools

    // 1. Weather tool (synchronous)
    let get_weather = function_tool(|request: WeatherRequest| {
        println!("[debug] get_weather called for {}", request.city);

        // In a real implementation, this would call a weather API
        WeatherResponse {
            temperature: 22.5,
            conditions: "Sunny".to_string(),
            humidity: 65,
        }
    });

    // 2. Restaurant recommendation tool (asynchronous)
    let get_restaurants = async_function_tool(|request: RestaurantRequest| async move {
        println!("[debug] get_restaurants called for {}", request.city);

        // In a real implementation, this would call a restaurant API
        let mut recommendations = vec![
            Restaurant {
                name: "Pasta Paradise".to_string(),
                cuisine: "Italian".to_string(),
                rating: 4.5,
            },
            Restaurant {
                name: "Sushi Supreme".to_string(),
                cuisine: "Japanese".to_string(),
                rating: 4.7,
            },
            Restaurant {
                name: "Taco Temple".to_string(),
                cuisine: "Mexican".to_string(),
                rating: 4.2,
            }
        ];

        // Filter by cuisine if specified
        if let Some(cuisine) = request.cuisine {
            recommendations.retain(|r| r.cuisine.to_lowercase().contains(&cuisine.to_lowercase()));
        }

        RestaurantResponse {
            recommendations,
        }
    });

    // Create an agent with both tools
    let agent: Agent = Agent::new(
        "Travel Assistant",
        "You are a helpful travel assistant. When asked about weather, use the get_weather tool. \
         When asked about restaurants, use the get_restaurants tool. Be concise but informative."
    )
        .with_tool(get_weather)
        .with_tool(get_restaurants);

    // Run the agent with a query about both weather and restaurants
    let result = Runner::run(
        &agent,
        "I'm going to Tokyo tomorrow. What's the weather like and can you recommend some Japanese restaurants?",
        None,
        None
    ).await?;

    // Print the result
    println!("{}", result.final_output);

    Ok(())
}

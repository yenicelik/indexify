use anyhow::{Error, Result};
use clap::{Parser, Subcommand};
use indexify::{CoordinatorServer, ExecutorServer, ServerConfig};
use opentelemetry::global;
use opentelemetry::sdk::Resource;
use opentelemetry::KeyValue;
use std::sync::Arc;
use tracing::{debug, info};
use tracing_subscriber::prelude::__tracing_subscriber_SubscriberExt;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(name = "indexify")]
#[command(about = "CLI for the Indexify Server", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    #[command(about = "Start the server")]
    StartServer {
        #[arg(short, long)]
        config_path: String,

        #[arg(short, long)]
        dev_mode: bool,
    },
    Coordinator {
        #[arg(short, long)]
        config_path: String,
    },
    Executor {
        #[arg(short, long)]
        config_path: String,
    },
    InitConfig {
        config_path: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    // Parse CLI and any env variables
    let args = Cli::parse();
    let version = format!(
        "git branch: {} - sha:{}",
        env!("VERGEN_GIT_BRANCH"),
        env!("VERGEN_GIT_SHA")
    );
    let filter = EnvFilter::from_default_env();

    // TODO: Traces should also be piped to stdout, not only tonic
    // Implement OpenTelemetry Tracer
    let otlp_exporter = opentelemetry_otlp::new_exporter().http();
    let tracer = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(otlp_exporter)
        .with_trace_config(
            opentelemetry::sdk::trace::config()
                .with_resource(Resource::new(vec![KeyValue::new(
                    "service.name",
                    // TODO: @diptanu, we should probably set a service-name here, could strumify the Commands enum (i.e. make each option also return a string)
                    "indexify-service",
                )]))
                // TODO: In production, we can change this config
                .with_sampler(opentelemetry::sdk::trace::Sampler::AlwaysOn),
        )
        .with_batch_config(opentelemetry::sdk::trace::BatchConfig::default())
        .install_batch(opentelemetry_sdk::runtime::Tokio)?;
    let otlp_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    // Hook it up to tracing
    let subscriber = tracing_subscriber::registry().with(filter).with(otlp_layer);
    tracing::subscriber::set_global_default(subscriber)?;

    // TODO: We can add another global span here, before we call the individual functions
    // This will be required to trace functions that are not explicity marked as #[instrument]

    // Start application
    info!("Spinning up");
    match args.command {
        Commands::StartServer {
            config_path,
            dev_mode,
        } => {
            info!("starting indexify server....");
            info!("version: {}", version);

            let config = indexify::ServerConfig::from_path(&config_path)?;
            debug!("Server config is: {:?}", config);
            let server = indexify::Server::new(Arc::new(config.clone()))?;
            debug!("Server struct is: {:?}", config);
            let server_handle = tokio::spawn(async move {
                server.run().await.unwrap();
            });
            if dev_mode {
                let coordinator = CoordinatorServer::new(Arc::new(config.clone())).await?;
                let coordinator_handle = tokio::spawn(async move {
                    coordinator.run().await.unwrap();
                });

                let executor_server = ExecutorServer::new(Arc::new(config.clone())).await?;
                let executor_handle = tokio::spawn(async move {
                    executor_server.run().await.unwrap();
                });
                tokio::try_join!(server_handle, coordinator_handle, executor_handle)?;
                return Ok(());
            }
            tokio::try_join!(server_handle)?;
        }
        Commands::InitConfig { config_path } => {
            println!("Initializing config file at: {}", &config_path);
            indexify::ServerConfig::generate(config_path).unwrap();
        }
        Commands::Coordinator { config_path } => {
            info!("starting indexify coordinator....");
            info!("version: {}", version);

            let config = ServerConfig::from_path(&config_path)?;
            let coordinator = CoordinatorServer::new(Arc::new(config)).await?;
            coordinator.run().await?
        }
        Commands::Executor { config_path } => {
            info!("starting indexify executor....");
            info!("version: {}", version);

            let config = ServerConfig::from_path(&config_path)?;
            let executor_server = ExecutorServer::new(Arc::new(config)).await?;
            executor_server.run().await?
        }
    }

    // Is shutdown here? Should probably also be configured on drop?
    global::shutdown_tracer_provider();
    Ok(())
}

""
Metrics Collector Service

Collects and stores system and trading metrics for monitoring and analysis.
"""
import time
import psutil
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import asyncio
import aiohttp
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects system and application metrics for monitoring."""
    
    def __init__(self, db_url: str, interval: int = 5):
        """Initialize the metrics collector.
        
        Args:
            db_url: Database connection URL
            interval: Collection interval in seconds
        """
        self.db_url = db_url
        self.interval = interval
        self.running = False
        self.session = None
        
    async def start(self):
        """Start the metrics collection loop."""
        self.running = True
        self.session = aiohttp.ClientSession()
        
        logger.info("Starting metrics collector...")
        while self.running:
            try:
                # Collect metrics
                metrics = await self.collect_metrics()
                
                # Store metrics
                await self.store_metrics(metrics)
                
                # Wait for next interval
                await asyncio.sleep(self.interval)
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(1)  # Prevent tight loop on errors
    
    async def stop(self):
        """Stop the metrics collector."""
        self.running = False
        if self.session:
            await self.session.close()
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect system and application metrics.
        
        Returns:
            Dictionary containing collected metrics
        """
        timestamp = datetime.utcnow().isoformat()
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network metrics
        net_io = psutil.net_io_counters()
        
        # Process metrics
        process = psutil.Process()
        process_mem = process.memory_info()
        
        return {
            'timestamp': timestamp,
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024 ** 3),
                'memory_total_gb': memory.total / (1024 ** 3),
                'disk_percent': disk.percent,
                'disk_used_gb': disk.used / (1024 ** 3),
                'disk_total_gb': disk.total / (1024 ** 3),
                'net_bytes_sent': net_io.bytes_sent,
                'net_bytes_recv': net_io.bytes_recv,
            },
            'process': {
                'pid': process.pid,
                'cpu_percent': process.cpu_percent(),
                'memory_rss_gb': process_mem.rss / (1024 ** 3),
                'memory_percent': process.memory_percent(),
                'num_threads': process.num_threads(),
                'num_fds': process.num_fds() if hasattr(process, 'num_fds') else 0,
            },
            'trading': await self.collect_trading_metrics()
        }
    
    async def collect_trading_metrics(self) -> Dict[str, Any]:
        """Collect trading-specific metrics.
        
        Returns:
            Dictionary containing trading metrics
        """
        # TODO: Implement collection of trading-specific metrics
        # This is a placeholder - replace with actual implementation
        return {
            'open_positions': 0,
            'total_pnl': 0.0,
            'daily_pnl': 0.0,
            'order_count': 0,
            'fill_rate': 0.0,
            'latency_ms': 0.0
        }
    
    async def store_metrics(self, metrics: Dict[str, Any]):
        """Store collected metrics in the database.
        
        Args:
            metrics: Dictionary of metrics to store
        """
        try:
            # System metrics
            system = metrics['system']
            trading = metrics['trading']
            
            # Prepare data for database
            data = {
                'timestamp': metrics['timestamp'],
                'cpu_usage': system['cpu_percent'],
                'memory_usage': system['memory_percent'],
                'disk_usage': system['disk_percent'],
                'network_sent': system['net_bytes_sent'],
                'network_recv': system['net_bytes_recv'],
                'open_positions': trading['open_positions'],
                'total_pnl': trading['total_pnl'],
                'order_count': trading['order_count'],
                'latency_ms': trading['latency_ms']
            }
            
            # In a real implementation, this would write to a time-series database
            # For example, using InfluxDB, TimescaleDB, or similar
            logger.debug(f"Storing metrics: {json.dumps(data, indent=2)}")
            
            # Example: Write to PostgreSQL
            async with self.session.post(
                f"{self.db_url}/api/ingest/metrics",
                json=data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    logger.error(f"Failed to store metrics: {await response.text()}")
        
        except Exception as e:
            logger.error(f"Error storing metrics: {e}")

class AlertManager:
    """Manages alerts based on collected metrics."""
    
    def __init__(self, db_url: str):
        """Initialize the alert manager.
        
        Args:
            db_url: Database connection URL
        """
        self.db_url = db_url
        self.alerts = {}
        self.session = None
    
    async def start(self):
        """Start the alert manager."""
        self.session = aiohttp.ClientSession()
        logger.info("Alert manager started")
    
    async def stop(self):
        """Stop the alert manager."""
        if self.session:
            await self.session.close()
    
    async def check_metrics(self, metrics: Dict[str, Any]):
        """Check metrics against alert rules.
        
        Args:
            metrics: Dictionary of metrics to check
        """
        alerts = []
        
        # System alerts
        system = metrics['system']
        if system['cpu_percent'] > 90:
            alerts.append({
                'timestamp': metrics['timestamp'],
                'severity': 'high',
                'message': f"High CPU usage: {system['cpu_percent']:.1f}%",
                'metric': 'cpu_percent',
                'value': system['cpu_percent']
            })
            
        if system['memory_percent'] > 90:
            alerts.append({
                'timestamp': metrics['timestamp'],
                'severity': 'high',
                'message': f"High memory usage: {system['memory_percent']:.1f}%",
                'metric': 'memory_percent',
                'value': system['memory_percent']
            })
        
        # Trading alerts
        trading = metrics['trading']
        if trading['latency_ms'] > 100:  # 100ms latency threshold
            alerts.append({
                'timestamp': metrics['timestamp'],
                'severity': 'medium',
                'message': f"High order latency: {trading['latency_ms']:.1f}ms",
                'metric': 'latency_ms',
                'value': trading['latency_ms']
            })
        
        # Process alerts
        for alert in alerts:
            await self.trigger_alert(alert)
    
    async def trigger_alert(self, alert: Dict[str, Any]):
        """Trigger an alert.
        
        Args:
            alert: Alert details
        """
        logger.warning(f"ALERT: {alert['message']}")
        
        # In a real implementation, this would:
        # 1. Store the alert in a database
        # 2. Send notifications (email, SMS, Slack, etc.)
        # 3. Trigger automated responses if needed
        
        try:
            async with self.session.post(
                f"{self.db_url}/api/alerts",
                json=alert,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    logger.error(f"Failed to send alert: {await response.text()}")
        except Exception as e:
            logger.error(f"Error sending alert: {e}")

class MonitoringService:
    """Main monitoring service that coordinates metrics collection and alerting."""
    
    def __init__(self, db_url: str, interval: int = 5):
        """Initialize the monitoring service.
        
        Args:
            db_url: Database connection URL
            interval: Metrics collection interval in seconds
        """
        self.metrics_collector = MetricsCollector(db_url, interval)
        self.alert_manager = AlertManager(db_url)
        self.tasks = []
    
    async def start(self):
        """Start the monitoring service."""
        logger.info("Starting monitoring service...")
        
        # Start components
        await self.alert_manager.start()
        
        # Start metrics collection in the background
        self.tasks.append(asyncio.create_task(self._run_metrics_loop()))
        
        logger.info("Monitoring service started")
    
    async def stop(self):
        """Stop the monitoring service."""
        logger.info("Stopping monitoring service...")
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Stop components
        await self.metrics_collector.stop()
        await self.alert_manager.stop()
        
        logger.info("Monitoring service stopped")
    
    async def _run_metrics_loop(self):
        """Run the metrics collection loop."""
        while True:
            try:
                # Collect metrics
                metrics = await self.metrics_collector.collect_metrics()
                
                # Store metrics
                await self.metrics_collector.store_metrics(metrics)
                
                # Check for alerts
                await self.alert_manager.check_metrics(metrics)
                
                # Wait for next interval
                await asyncio.sleep(self.metrics_collector.interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics loop: {e}")
                await asyncio.sleep(1)  # Prevent tight loop on errors

# Example usage
async def main():
    """Example usage of the monitoring service."""
    # In a real application, this would come from configuration
    DB_URL = "http://localhost:8086"  # Example: InfluxDB URL
    
    # Create and start the monitoring service
    monitoring = MonitoringService(DB_URL)
    await monitoring.start()
    
    try:
        # Keep the service running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        # Clean up on exit
        await monitoring.stop()

if __name__ == "__main__":
    asyncio.run(main())

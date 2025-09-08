// Updated DashboardHome with Translation Support
// src/components/DashboardHome.js

import React, { useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Grid,
  Box,
  Chip,
  Button,
  CircularProgress,
  Alert,
  Tabs,
  Tab
} from '@mui/material';
import {
  Circle as StatusIcon,
  Refresh as RefreshIcon,
  Thermostat,
  Water,
  WbSunny,
  Opacity,
  Cloud,
  Battery3Bar,
  Settings as SettingsIcon,
  Camera as CameraIcon,
  Warning as AlertsIcon,
  TrendingUp as TrendingUpIcon,
  CheckCircle as CheckCircleIcon
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useSensorData } from '../hooks/useSensorData';
import { useAuth } from '../context/AuthContext';
import { useTranslation } from '../context/LanguageContext'; // Add this import
import SensorChart from './dashboard/SensorChart';
import { useFirebaseChartData } from '../hooks/useFirebaseChartData';

function DashboardHome() {
  const { sensorData, loading, error, lastUpdated, refreshData } = useSensorData();
  const { userProfile } = useAuth();
  const { t } = useTranslation(); // Add translation hook
  const navigate = useNavigate();
  const [selectedChart, setSelectedChart] = useState('temperature');

  // Get chart data from Firebase
  const { chartData, loading: chartLoading, error: chartError } = useFirebaseChartData(selectedChart);

  // Generate chart configuration based on selected sensor
  const getChartConfig = (sensorType) => {
    const configs = {
      temperature: {
        title: t('temperatureTrend'),
        dataKeys: ['temperature'],
        colors: ['#ef4444'],
        unit: ['°C']
      },
      humidity: {
        title: t('humidityTrend'), 
        dataKeys: ['humidity'],
        colors: ['#3b82f6'],
        unit: ['%']
      },
      soilMoisture: {
        title: t('soilMoistureTrend'),
        dataKeys: ['soilMoisture'],
        colors: ['#10b981'],
        unit: ['%']
      },
      lightIntensity: {
        title: t('lightIntensityTrend'),
        dataKeys: ['lightIntensity'],
        colors: ['#f59e0b'],
        unit: ['Lux']
      },
      rainSensor: {
        title: t('rainSensorTrend'),
        dataKeys: ['rainSensor'],
        colors: ['#8b5cf6'],
        unit: ['%']
      }
    };

    return configs[sensorType] || configs.temperature;
  };

// Update the SensorCard component in DashboardHome.js
// Replace the existing SensorCard component with this updated version:

  const SensorCard = ({ title, data, icon: Icon }) => {
    const { formatSensorValue } = useTranslation(); // Add this line at the top of the component
    
    if (!data) return null;

    return (
      <Card 
        elevation={2}
        style={{ 
          height: '160px', 
          border: `2px solid ${data.status?.toLowerCase() === 'good' || data.status?.toLowerCase() === 'active' ? '#22c55e' : data.status?.toLowerCase() === 'warning' ? '#f59e0b' : '#ef4444'}`,
          borderRadius: '12px',
          transition: 'all 0.3s ease',
          backgroundColor: 'white'
        }}
      >
        <CardContent style={{ padding: '1.25rem' }}>
          <Box display="flex" alignItems="center" justifyContent="space-between" mb={1.5}>
            <Icon style={{ 
              fontSize: '2rem', 
              color: data.status?.toLowerCase() === 'good' || data.status?.toLowerCase() === 'active' ? '#22c55e' : data.status?.toLowerCase() === 'warning' ? '#f59e0b' : '#ef4444'
            }} />
            <Chip 
              label={t(data.status?.toLowerCase()) || data.status} 
              size="small" 
              style={{ 
                backgroundColor: data.status?.toLowerCase() === 'good' || data.status?.toLowerCase() === 'active' ? '#dcfce7' : 
                              data.status?.toLowerCase() === 'warning' ? '#fef3c7' : '#fecaca',
                color: data.status?.toLowerCase() === 'good' || data.status?.toLowerCase() === 'active' ? '#166534' : 
                      data.status?.toLowerCase() === 'warning' ? '#92400e' : '#991b1b',
                fontWeight: 'bold'
              }} 
            />
          </Box>
          <Typography variant="h6" style={{ fontWeight: 600, marginBottom: '0.75rem', color: '#1f2937' }}>
            {title}
          </Typography>
          <Typography variant="h4" style={{ 
            fontWeight: 'bold', 
            color: data.status?.toLowerCase() === 'good' || data.status?.toLowerCase() === 'active' ? '#22c55e' : data.status?.toLowerCase() === 'warning' ? '#f59e0b' : '#ef4444',
            marginBottom: '0.75rem'
          }}>
            {/* Updated number formatting */}
            {title === t('activeBattery') 
              ? formatSensorValue(data.value, 0) // No decimals for battery count
              : formatSensorValue(data.value) // Format with appropriate decimals
            }
            <span style={{ fontSize: '1rem', marginLeft: '0.25rem' }}>{data.unit}</span>
          </Typography>
          
          {/* Status indicator bar */}
          <Box>
            <div style={{ width: '100%', backgroundColor: '#f3f4f6', borderRadius: '4px', height: '4px' }}>
              <div
                style={{
                  height: '4px',
                  borderRadius: '4px',
                  backgroundColor: data.status?.toLowerCase() === 'good' || data.status?.toLowerCase() === 'active' ? '#22c55e' : 
                                  data.status?.toLowerCase() === 'warning' ? '#f59e0b' : '#ef4444',
                  width: `${data.status?.toLowerCase() === 'good' || data.status?.toLowerCase() === 'active' ? '90%' : 
                          data.status?.toLowerCase() === 'warning' ? '65%' : '35%'}`,
                  transition: 'all 0.3s ease'
                }}
              />
            </div>
          </Box>
        </CardContent>
      </Card>
    );
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="60vh">
        <CircularProgress size={60} style={{ color: '#22c55e' }} />
        <Typography variant="h6" ml={2} style={{ color: '#1f2937' }}>
          {t('loadingSensorData')}
        </Typography>
      </Box>
    );
  }

  if (error && !sensorData) {
    return (
      <Alert 
        severity="error" 
        action={<Button color="inherit" size="small" onClick={refreshData}>{t('refresh')}</Button>}
        style={{ backgroundColor: '#fef2f2', color: '#991b1b', border: '1px solid #fca5a5' }}
      >
        {t('errorLoadingSensorData')}: {error}
      </Alert>
    );
  }

  const activeSensors = sensorData ? Object.keys(sensorData).filter(key => 
    !['diseaseRisks', 'timestamp'].includes(key)).length : 0;
  const alertsCount = sensorData ? Object.values(sensorData).filter(sensor => 
    sensor?.status === 'warning' || sensor?.status === 'critical').length : 0;

  const chartConfig = getChartConfig(selectedChart);

  return (
    <div style={{ padding: '0', backgroundColor: '#f8fafc' }}>
      {/* Top Section - All in One Row with Equal Spacing */}
      <Box style={{ display: 'flex', gap: '1rem', marginBottom: '2rem', flexWrap: 'wrap' }}>
        {/* Connected Status */}
        <Card elevation={1} style={{ backgroundColor: 'white', borderRadius: '12px', flex: '1', minWidth: '200px', minHeight: '140px' }}>
          <CardContent style={{ padding: '1.25rem', height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
            <Box>
              <Box display="flex" alignItems="center" mb={1}>
                <StatusIcon style={{ color: '#22c55e', marginRight: '0.5rem', fontSize: '1.25rem' }} />
                <Typography variant="body1" style={{ fontWeight: 600, color: '#1f2937', fontSize: '0.9rem' }}>
                  {t('connectedToSensors')}
                </Typography>
              </Box>
              <Typography variant="body2" color="textSecondary" style={{ marginBottom: '0.75rem' }}>
                {t('lastUpdated')}: {lastUpdated ? lastUpdated.toLocaleDateString() + ' ' + lastUpdated.toLocaleTimeString() : 'Loading...'}
              </Typography>
            </Box>
            <Button 
              startIcon={<RefreshIcon />} 
              onClick={refreshData}
              disabled={loading}
              size="small"
              style={{ color: '#22c55e', borderColor: '#22c55e' }}
              variant="outlined"
            >
              {t('refresh')}
            </Button>
          </CardContent>
        </Card>

        {/* Disease Risk */}
        <Card elevation={1} style={{ backgroundColor: 'white', borderRadius: '12px', border: '2px solid #22c55e', flex: '1', minWidth: '200px', minHeight: '140px' }}>
          <CardContent style={{ padding: '1.25rem', height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Box display="flex" alignItems="center" mb={1}>
              <CheckCircleIcon style={{ color: '#22c55e', marginRight: '0.5rem', fontSize: '1.25rem' }} />
              <Typography variant="body1" style={{ fontWeight: 600, color: '#1f2937', fontSize: '0.9rem' }}>
                {t('diseaseRiskStatus')}
              </Typography>
            </Box>
            <Typography variant="body2" color="textSecondary">
              {t('noImmediateRisks')}
            </Typography>
          </CardContent>
        </Card>

        {/* Farm Health */}
        <Card elevation={1} style={{ backgroundColor: 'white', borderRadius: '12px', border: '2px solid #22c55e', flex: '1', minWidth: '200px', minHeight: '140px' }}>
          <CardContent style={{ padding: '1.25rem', textAlign: 'center', height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <Typography variant="h6" style={{ fontWeight: 600, marginBottom: '0.75rem', color: '#1f2937' }}>
              {t('farmHealth')}
            </Typography>
            <Typography variant="h3" style={{ fontWeight: 'bold', color: '#22c55e', marginBottom: '0.5rem' }}>
              95%
            </Typography>
            <Typography variant="body1" style={{ color: '#22c55e' }}>
              {t('excellent')}
            </Typography>
          </CardContent>
        </Card>

        {/* Quick Actions */}
        <Card elevation={1} style={{ backgroundColor: 'white', borderRadius: '12px', flex: '1', minWidth: '200px', minHeight: '140px' }}>
          <CardContent style={{ padding: '1.25rem', height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Typography variant="h6" style={{ fontWeight: 600, marginBottom: '0.75rem', color: '#1f2937' }}>
              {t('quickActions')}
            </Typography>
            <Button
              onClick={() => navigate('/detection')}
              style={{ 
                padding: '6px 12px',
                backgroundColor: '#faf5ff',
                color: '#8b5cf6',
                textTransform: 'none',
                marginBottom: '6px',
                width: '100%'
              }}
              startIcon={<CameraIcon />}
              size="small"
            >
              {t('diseaseDetection')}
            </Button>
            <Button
              onClick={() => navigate('/alerts')}
              style={{ 
                padding: '6px 12px',
                backgroundColor: '#fff7ed',
                color: '#f59e0b',
                textTransform: 'none',
                width: '100%'
              }}
              startIcon={<AlertsIcon />}
              size="small"
            >
              {t('viewAlerts')} ({alertsCount})
            </Button>
          </CardContent>
        </Card>
      </Box>

      {/* Main Environmental Sensors - 5 in full width */}
      <Box display="flex" alignItems="center" style={{ marginBottom: '1.5rem' }}>
        <TrendingUpIcon style={{ color: '#22c55e', marginRight: '0.5rem', fontSize: '1.5rem' }} />
        <Typography variant="h5" style={{ fontWeight: 600, color: '#1f2937' }}>
          {t('environmentalStatus')}
        </Typography>
      </Box>

      <Grid container spacing={3} style={{ marginBottom: '2rem' }}>
        {sensorData && (
          <>
            <Grid item xs={12} sm={6} md={2.4}>
              <SensorCard title={t('soilMoisture')} data={sensorData.soilMoisture} icon={Water} />
            </Grid>
            <Grid item xs={12} sm={6} md={2.4}>
              <SensorCard title={t('temperature')} data={sensorData.temperature} icon={Thermostat} />
            </Grid>
            <Grid item xs={12} sm={6} md={2.4}>
              <SensorCard title={t('humidity')} data={sensorData.humidity} icon={Opacity} />
            </Grid>
            <Grid item xs={12} sm={6} md={2.4}>
              <SensorCard title={t('lightIntensity')} data={sensorData.lightIntensity} icon={WbSunny} />
            </Grid>
            <Grid item xs={12} sm={6} md={2.4}>
              <SensorCard title={t('rainSensor')} data={sensorData.rainSensor} icon={Cloud} />
            </Grid>
          </>
        )}
      </Grid>

      {/* Additional Sensors - 2 sensors aligned left */}
      <Typography variant="h6" style={{ fontWeight: 600, color: '#6b7280', marginBottom: '1.5rem' }}>
        {t('additionalSensors')}
      </Typography>

      <Grid container spacing={3} style={{ marginBottom: '2rem' }}>
        {sensorData && (
          <>
            <Grid item xs={12} sm={6} md={2.4}>
              <SensorCard title={t('activeBattery')} data={sensorData.activeBattery} icon={Battery3Bar} />
            </Grid>
            <Grid item xs={12} sm={6} md={2.4}>
              <SensorCard title={t('batteryVoltage')} data={sensorData.batteryVoltage} icon={SettingsIcon} />
            </Grid>
          </>
        )}
      </Grid>

      {/* Interactive Sensor Trends with Real Firebase Data */}
      <Box display="flex" alignItems="center" style={{ marginBottom: '1.5rem' }}>
        <TrendingUpIcon style={{ color: '#22c55e', marginRight: '0.5rem', fontSize: '1.5rem' }} />
        <Typography variant="h5" style={{ fontWeight: 600, color: '#1f2937' }}>
          {t('sensorTrends')}
        </Typography>
      </Box>

      <Card elevation={2} style={{ backgroundColor: 'white', borderRadius: '12px' }}>
        <CardContent style={{ padding: '2rem' }}>
          {/* Sensor Selection Tabs */}
          <Tabs
            value={selectedChart}
            onChange={(e, newValue) => setSelectedChart(newValue)}
            style={{ marginBottom: '1.5rem' }}
            indicatorColor="primary"
          >
            <Tab 
              label={t('temperature')} 
              value="temperature" 
              style={{ textTransform: 'none', fontWeight: 600, color: '#1f2937' }}
            />
            <Tab 
              label={t('humidity')} 
              value="humidity" 
              style={{ textTransform: 'none', fontWeight: 600, color: '#1f2937' }}
            />
            <Tab 
              label={t('soilMoisture')} 
              value="soilMoisture" 
              style={{ textTransform: 'none', fontWeight: 600, color: '#1f2937' }}
            />
            <Tab 
              label={t('lightIntensity')} 
              value="lightIntensity" 
              style={{ textTransform: 'none', fontWeight: 600, color: '#1f2937' }}
            />
            <Tab 
              label={t('rainSensor')} 
              value="rainSensor" 
              style={{ textTransform: 'none', fontWeight: 600, color: '#1f2937' }}
            />
          </Tabs>

          {/* Chart with Real Firebase Data */}
          <Box style={{ height: '400px', width: '100%' }}>
            {chartLoading ? (
              <Box display="flex" justifyContent="center" alignItems="center" height="100%">
                <CircularProgress style={{ color: '#22c55e' }} />
                <Typography variant="body1" ml={2}>{t('loadingChartData')}</Typography>
              </Box>
            ) : chartError ? (
              <Box display="flex" justifyContent="center" alignItems="center" height="100%">
                <Alert severity="error">
                  Error loading chart: {chartError}
                </Alert>
              </Box>
            ) : (
              <SensorChart
                title={chartConfig.title}
                data={chartData}
                dataKeys={chartConfig.dataKeys}
                colors={chartConfig.colors}
                unit={chartConfig.unit}
              />
            )}
          </Box>
        </CardContent>
      </Card>
    </div>
  );
}

export default DashboardHome;
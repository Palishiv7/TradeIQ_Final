'use client';

import React, { useRef, useEffect, useState } from 'react';
import { createChart, IChartApi, ISeriesApi, CandlestickData, CrosshairMode, ChartOptions, ColorType, Time } from 'lightweight-charts';

interface ChartData {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

interface ExtendedCandlestickData extends CandlestickData<Time> {
  volume?: number;
}

interface CandlestickChartProps {
  data: ChartData[];
  width?: number;
  height?: number;
  darkMode?: boolean;
  showVolume?: boolean;
  showGrid?: boolean;
  showCrosshair?: boolean;
  showTooltip?: boolean;
  showLegend?: boolean;
  toolTipContent?: React.ReactNode;
  onCrosshairMove?: (param: any) => void;
  onBarHover?: (param: ExtendedCandlestickData | null) => void;
  className?: string;
  colors?: {
    backgroundColor?: string;
    lineColor?: string;
    textColor?: string;
    areaTopColor?: string;
    areaBottomColor?: string;
  };
}

export default function CandlestickChart({
  data,
  width,
  height = 400,
  darkMode = false,
  showVolume = true,
  showGrid = true,
  showCrosshair = true,
  showTooltip = true,
  showLegend = true,
  toolTipContent,
  onCrosshairMove,
  onBarHover,
  className,
  colors = {},
}: CandlestickChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const [chartCreated, setChartCreated] = useState<IChartApi | null>(null);
  const [candleSeries, setCandleSeries] = useState<ISeriesApi<'Candlestick'> | null>(null);
  const [volumeSeries, setVolumeSeries] = useState<ISeriesApi<'Histogram'> | null>(null);
  const [hoveredBar, setHoveredBar] = useState<ExtendedCandlestickData | null>(null);

  // Create chart on component mount
  useEffect(() => {
    if (chartContainerRef.current) {
      const container = chartContainerRef.current;
      const containerWidth = width || container.clientWidth;
      const containerHeight = height;

      // Clear container
      container.innerHTML = '';

      // Chart options
      const options: ChartOptions = {
        width: containerWidth,
        height: containerHeight,
        layout: {
          background: {
            type: ColorType.Solid,
            color: darkMode ? '#1E1E1E' : '#FFFFFF',
          },
          textColor: darkMode ? '#D9D9D9' : '#191919',
          fontSize: 12,
          fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
          attributionLogo: false,
        },
        localization: {
          locale: 'en-US',
          dateFormat: 'yyyy-MM-dd',
        },
        autoSize: true,
        watermark: {
          visible: false,
          color: 'rgba(0, 0, 0, 0)',
          text: '',
          fontSize: 12,
          fontFamily: '-apple-system',
          fontStyle: '',
          horzAlign: 'center',
          vertAlign: 'center',
        },
        leftPriceScale: {
          visible: false,
          autoScale: true,
          mode: 0,
          invertScale: false,
          alignLabels: true,
          borderVisible: true,
          entireTextOnly: true,
          ticksVisible: true,
          scaleMargins: {
            top: 0.2,
            bottom: 0.2,
          },
          borderColor: darkMode ? '#606060' : '#C8C8C8',
          minimumWidth: 30,
        },
        grid: {
          horzLines: {
            color: darkMode ? '#404040' : '#E6E6E6',
            style: 1,
            visible: showGrid,
          },
          vertLines: {
            color: darkMode ? '#404040' : '#E6E6E6',
            style: 1,
            visible: showGrid,
          },
        },
        crosshair: {
          mode: showCrosshair ? CrosshairMode.Normal : CrosshairMode.Magnet,
          vertLine: {
            width: 1,
            color: '#606060',
            style: 0,
            visible: showCrosshair,
            labelVisible: true,
            labelBackgroundColor: darkMode ? '#1E1E1E' : '#FFFFFF',
          },
          horzLine: {
            width: 1,
            color: '#606060',
            style: 0,
            visible: showCrosshair,
            labelVisible: true,
            labelBackgroundColor: darkMode ? '#1E1E1E' : '#FFFFFF',
          },
        },
        timeScale: {
          borderColor: darkMode ? '#606060' : '#C8C8C8',
          timeVisible: true,
          secondsVisible: false,
          rightOffset: 12,
          barSpacing: 6,
          minBarSpacing: 4,
          fixLeftEdge: true,
          fixRightEdge: true,
          lockVisibleTimeRangeOnResize: true,
          rightBarStaysOnScroll: true,
          borderVisible: true,
          visible: true,
          ticksVisible: true,
          uniformDistribution: true,
          shiftVisibleRangeOnNewBar: true,
          allowShiftVisibleRangeOnWhitespaceReplacement: true,
          minimumHeight: 0,
          allowBoldLabels: true,
        },
        rightPriceScale: {
          borderColor: darkMode ? '#606060' : '#C8C8C8',
          autoScale: true,
          mode: 0,
          invertScale: false,
          alignLabels: true,
          borderVisible: true,
          visible: true,
          entireTextOnly: true,
          ticksVisible: true,
          minimumWidth: 30,
          scaleMargins: {
            top: 0.2,
            bottom: 0.2,
          },
        },
        handleScroll: {
          mouseWheel: true,
          pressedMouseMove: true,
          horzTouchDrag: true,
          vertTouchDrag: true,
        },
        handleScale: {
          axisPressedMouseMove: {
            time: true,
            price: true,
          },
          axisDoubleClickReset: true,
          mouseWheel: true,
          pinch: true,
        },
        kineticScroll: {
          mouse: true,
          touch: true,
        },
        trackingMode: {
          exitMode: 1,
        },
        overlayPriceScales: {
          borderVisible: true,
          mode: 0,
          invertScale: false,
          alignLabels: true,
          scaleMargins: {
            top: 0.2,
            bottom: 0.2,
          },
          entireTextOnly: true,
          borderColor: darkMode ? '#606060' : '#C8C8C8',
          ticksVisible: true,
          minimumWidth: 30,
        },
      };

      // Create chart
      const chart = createChart(container, options);
      setChartCreated(chart);

      // Add candlestick series
      const candlestickSeries = chart.addCandlestickSeries({
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderVisible: false,
        wickUpColor: '#26a69a',
        wickDownColor: '#ef5350',
      });
      setCandleSeries(candlestickSeries);

      // Add volume series if enabled
      if (showVolume) {
        const volumeSeriesInstance = chart.addHistogramSeries({
          color: '#26a69a',
          priceFormat: {
            type: 'volume',
          },
          priceScaleId: '',
        });
        setVolumeSeries(volumeSeriesInstance);
      }

      // Setup event handlers
      if (showCrosshair && onCrosshairMove) {
        chart.subscribeCrosshairMove((param) => {
          onCrosshairMove(param);

          // Extract hovered bar data
          if (param && param.time) {
            const hoveredData = data.find((d) => d.time === param.time);
            if (hoveredData) {
              setHoveredBar(hoveredData);
              if (onBarHover) onBarHover(hoveredData);
            }
          } else {
            setHoveredBar(null);
            if (onBarHover) onBarHover(null);
          }
        });
      }

      // Convert string dates to timestamps for the chart
      if (Array.isArray(data) && data.length > 0) {
        const chartData = data.map(item => ({
          time: new Date(item.time).getTime() / 1000 as Time,
          open: item.open,
          high: item.high,
          low: item.low,
          close: item.close,
        }));

        // Set initial data
        candlestickSeries.setData(chartData);
        
        // Set volume data if enabled
        if (showVolume && volumeSeries) {
          const volumeData = data.map(item => ({
            time: new Date(item.time).getTime() / 1000 as Time,
            value: item.volume || 0,
            color: (item.close || 0) >= (item.open || 0) ? '#26a69a' : '#ef5350',
          }));
          volumeSeries.setData(volumeData);
        }
      }

      // Handle window resize
      const handleResize = () => {
        if (container) {
          const newWidth = width || container.clientWidth;
          chart.applyOptions({ width: newWidth, height });
          chart.timeScale().fitContent();
        }
      };

      window.addEventListener('resize', handleResize);

      // Cleanup function
      return () => {
        window.removeEventListener('resize', handleResize);
        chart.remove();
        setChartCreated(null);
        setCandleSeries(null);
        setVolumeSeries(null);
      };
    }
  }, [
    data,
    width,
    height,
    darkMode,
    showVolume,
    showGrid,
    showCrosshair,
    onCrosshairMove,
    onBarHover,
  ]);

  // Update data when it changes
  useEffect(() => {
    if (candleSeries && Array.isArray(data) && data.length > 0) {
      const chartData = data.map(item => ({
        time: new Date(item.time).getTime() / 1000 as Time,
        open: item.open,
        high: item.high,
        low: item.low,
        close: item.close,
      }));
      candleSeries.setData(chartData);
      
      if (showVolume && volumeSeries) {
        const volumeData = data.map(item => ({
          time: new Date(item.time).getTime() / 1000 as Time,
          value: item.volume || 0,
          color: (item.close || 0) >= (item.open || 0) ? '#26a69a' : '#ef5350',
        }));
        volumeSeries.setData(volumeData);
      }
      
      if (chartCreated) {
        chartCreated.timeScale().fitContent();
      }
    }
  }, [data, candleSeries, volumeSeries, chartCreated, showVolume]);

  return (
    <div className={`relative ${className || ''}`}>
      {/* Chart container */}
      <div 
        ref={chartContainerRef} 
        className="w-full h-full"
      />
      
      {/* Custom tooltip */}
      {showTooltip && hoveredBar && (
        <div className="absolute top-4 right-4 bg-white/90 dark:bg-gray-800/90 p-2 rounded shadow-md z-10 text-sm">
          {toolTipContent ? (
            toolTipContent
          ) : (
            <div>
              <div className="grid grid-cols-2 gap-x-4">
                <span className="font-medium">Open:</span>
                <span className="text-right">{hoveredBar.open}</span>
                <span className="font-medium">High:</span>
                <span className="text-right">{hoveredBar.high}</span>
                <span className="font-medium">Low:</span>
                <span className="text-right">{hoveredBar.low}</span>
                <span className="font-medium">Close:</span>
                <span className="text-right">{hoveredBar.close}</span>
                {hoveredBar.volume && (
                  <>
                    <span className="font-medium">Volume:</span>
                    <span className="text-right">{hoveredBar.volume.toLocaleString()}</span>
                  </>
                )}
              </div>
            </div>
          )}
        </div>
      )}
      
      {/* Legend */}
      {showLegend && (
        <div className="absolute top-0 left-0 p-2 bg-white/90 dark:bg-gray-800/90 rounded-br shadow-sm z-10 text-xs">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-[#26a69a]"></div>
            <span>Bullish</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-[#ef5350]"></div>
            <span>Bearish</span>
          </div>
        </div>
      )}
    </div>
  );
} 
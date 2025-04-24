
import React, { useState, useEffect } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ZAxis, Label } from 'recharts';
import Papa from 'papaparse';
import _ from 'lodash';

const YeastFitnessScatterPlot = () => {
  const [plotData, setPlotData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    const processAllFiles = async () => {
      try {
        setLoading(true);
        
        // Define the file names
        const fileNames = [
          'Clim_rep1_FitSeq2_Result.csv',
          'Clim_rep3_FitSeq2_Result.csv',
          'Nlim_rep1_FitSeq2_Result.csv',
          'Nlim_rep2_FitSeq2_Result.csv',
          'Nlim_rep3_FitSeq2_Result.csv',
          'Switch_rep1_FitSeq2_Result.csv',
          'Switch_rep2_FitSeq2_Result.csv',
          'Switch_rep3_FitSeq2_Result.csv'
        ];
        
        // Process files by condition
        const processByCondition = async (condition, files) => {
          const allDataByMutant = Array(3907).fill().map((_, index) => ({ 
            mutantIndex: index, 
            [condition]: [] 
          }));
          
          for (const fileName of files) {
            try {
              const fileContent = await window.fs.readFile(fileName, { encoding: 'utf8' });
              const parsed = Papa.parse(fileContent, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true
              });
              
              // Filter out mutants with Error_Fitness > 1
              const filteredData = parsed.data.filter(row => row.Error_Fitness <= 1);
              
              // Add data to the appropriate index
              filteredData.forEach((row, index) => {
                if (index < allDataByMutant.length) {
                  allDataByMutant[index][condition].push(row.Fitness_Per_Cycle);
                }
              });
            } catch (error) {
              console.error(`Error processing ${fileName}:`, error);
            }
          }
          
          return allDataByMutant;
        };
        
        // Process all conditions
        const climFiles = fileNames.filter(name => name.startsWith('Clim'));
        const nlimFiles = fileNames.filter(name => name.startsWith('Nlim'));
        
        const climDataByMutant = await processByCondition('clim', climFiles);
        const nlimDataByMutant = await processByCondition('nlim', nlimFiles);
        
        // Merge and calculate averages
        const mergedData = [];
        
        for (let i = 0; i < Math.max(climDataByMutant.length, nlimDataByMutant.length); i++) {
          const climData = climDataByMutant[i]?.clim;
          const nlimData = nlimDataByMutant[i]?.nlim;
          
          if (climData && climData.length > 0 && nlimData && nlimData.length > 0) {
            const avgClimFitness = _.mean(climData);
            const avgNlimFitness = _.mean(nlimData);
            
            mergedData.push({
              mutantIndex: i,
              climFitness: avgClimFitness,
              nlimFitness: avgNlimFitness,
              // Calculate combined average for color gradient
              combinedFitness: (avgClimFitness + avgNlimFitness) / 2
            });
          }
        }
        
        setPlotData(mergedData);
        setLoading(false);
      } catch (error) {
        console.error('Error processing data:', error);
        setError('Error processing data. Please check console for details.');
        setLoading(false);
      }
    };
    
    processAllFiles();
  }, []);
  
  // Calculate regression line for reference
  const calculateRegressionLine = (data) => {
    if (!data || data.length === 0) return null;
    
    // Create transformed data for the scatter plot
    const transformedData = data.map(point => ({
      x: point.nlimFitness,
      y: point.climFitness
    }));
    
    const xValues = transformedData.map(d => d.x);
    const yValues = transformedData.map(d => d.y);
    
    const xMean = _.mean(xValues);
    const yMean = _.mean(yValues);
    
    // Calculate slope (m) and y-intercept (b)
    let numerator = 0;
    let denominator = 0;
    
    for (let i = 0; i < transformedData.length; i++) {
      numerator += (xValues[i] - xMean) * (yValues[i] - yMean);
      denominator += Math.pow(xValues[i] - xMean, 2);
    }
    
    const slope = numerator / denominator;
    const intercept = yMean - slope * xMean;
    
    // Find the min and max x values for drawing the line
    const minX = Math.min(...xValues);
    const maxX = Math.max(...xValues);
    
    return [
      { x: minX, y: slope * minX + intercept },
      { x: maxX, y: slope * maxX + intercept }
    ];
  };
  
  const regressionLine = calculateRegressionLine(plotData);
  
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-2 border border-gray-300 shadow-md">
          <p className="text-sm font-semibold">{`Mutant Index: ${payload[0].payload.mutantIndex}`}</p>
          <p className="text-sm">{`Clim Fitness: ${payload[0].payload.climFitness.toFixed(4)}`}</p>
          <p className="text-sm">{`Nlim Fitness: ${payload[0].payload.nlimFitness.toFixed(4)}`}</p>
        </div>
      );
    }
    return null;
  };

  if (loading) {
    return <div className="flex justify-center items-center h-64">Loading data...</div>;
  }

  if (error) {
    return <div className="flex justify-center items-center h-64 text-red-500">{error}</div>;
  }

  return (
    <div className="flex flex-col items-center w-full">
      <h2 className="text-xl font-bold mb-4">Yeast Mutant Fitness: Clim vs Nlim</h2>
      <div className="w-full h-96">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart
            margin={{ top: 20, right: 20, bottom: 70, left: 70 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              type="number" 
              dataKey="x" 
              name="Nlim Fitness"
              domain={['auto', 'auto']}
            >
              <Label value="Fitness in Nlim" position="bottom" offset={20} />
            </XAxis>
            <YAxis 
              type="number" 
              dataKey="y" 
              name="Clim Fitness"
              domain={['auto', 'auto']}
            >
              <Label value="Fitness in Clim" angle={-90} position="left" offset={-40} />
            </YAxis>
            <ZAxis type="number" range={[20, 20]} />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            
            {/* Scatter plot of fitness values */}
            <Scatter 
              name="Mutant Fitness" 
              data={plotData.map(point => ({
                x: point.nlimFitness,
                y: point.climFitness,
                mutantIndex: point.mutantIndex,
                climFitness: point.climFitness,
                nlimFitness: point.nlimFitness
              }))} 
              fill="#8884d8"
              opacity={0.6}
            />
            
            {/* Diagonal line (y=x) for reference */}
            <Scatter
              name="y = x"
              data={[
                { x: -0.3, y: -0.3 },
                { x: 0.3, y: 0.3 }
              ]}
              line={{ stroke: '#ff7300', strokeWidth: 1, strokeDasharray: '5 5' }}
              shape={() => null}
              legendType="line"
            />
            
            {/* Regression line */}
            {regressionLine && (
              <Scatter
                name="Regression Line"
                data={regressionLine.map(point => ({ x: point.x, y: point.y }))}
                line={{ stroke: '#00C49F', strokeWidth: 2 }}
                shape={() => null}
                legendType="line"
              />
            )}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-4 text-sm text-gray-600">
        <p>Total mutants plotted: {plotData.length}</p>
        <p>Mutants filtered out: Error_Fitness > 1</p>
      </div>
    </div>
  );
};

export default YeastFitnessScatterPlot;
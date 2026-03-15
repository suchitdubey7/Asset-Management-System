import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, PieChart, Pie, Cell, BarChart, Bar } from 'recharts';
import './App.css';

function App() {
  const [report, setReport] = useState(null);
  const [portfolio, setPortfolio] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    const apiBase = process.env.REACT_APP_API_URL || 'http://localhost:8001';

    Promise.all([
      fetch(`${apiBase}/report`).then(response => response.json()),
      fetch(`${apiBase}/portfolio`).then(response => response.json()),
    ])
      .then(([reportJson, portfolioJson]) => {
        setReport(reportJson);
        setPortfolio(portfolioJson);
        setLoading(false);
      })
      .catch(error => {
        console.error('Error loading data:', error);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return <div className="flex items-center justify-center h-screen bg-gray-900 text-white">Loading...</div>;
  }

  if (!report || !portfolio) {
    return <div className="flex items-center justify-center h-screen bg-gray-900 text-white">No data available</div>;
  }

  const performance = report.performance;
  const risk = report.risk;
  const alerts = report.alerts || [];
  const macroRiskFactors = report.macro_risk_factors || [];
  const holdingData = portfolio.holdings || [];

  // Mock performance data for chart (replace with real if available)
  const chartData = [
    { name: 'Jan', value: 100 },
    { name: 'Feb', value: 120 },
    { name: 'Mar', value: 110 },
  ];

  const sectorExposureMap = {};
  holdingData.forEach(h => {
    sectorExposureMap[h.sector] = (sectorExposureMap[h.sector] || 0) + (h.weight || 0);
  });
  const sectorData = Object.entries(sectorExposureMap).map(([sector, exposure]) => ({
    name: sector,
    value: exposure * 100,
  }));

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

  const renderOverview = () => (
    <>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <div className="bg-gray-800 p-4 rounded">
          <h3 className="text-lg">Portfolio NAV</h3>
          <p className="text-2xl">₹{(portfolio.nav / 1e7).toFixed(0)} crores</p>
        </div>
        <div className="bg-gray-800 p-4 rounded">
          <h3 className="text-lg">YTD Return</h3>
          <p className={`text-2xl ${performance.ytd_return > 0 ? 'text-green-400' : 'text-red-400'}`}>
            {(performance.ytd_return * 100).toFixed(2)}%
          </p>
        </div>
        <div className="bg-gray-800 p-4 rounded">
          <h3 className="text-lg">Risk Level</h3>
          <p className="text-2xl">{risk.level}</p>
        </div>
        <div className="bg-gray-800 p-4 rounded">
          <h3 className="text-lg">VaR (1d)</h3>
          <p className="text-2xl">{(risk.var_1d * 100).toFixed(2)}%</p>
        </div>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-800 p-4 rounded">
          <h3 className="text-lg mb-4">Performance Chart</h3>
          <LineChart width={400} height={300} data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="value" stroke="#8884d8" />
          </LineChart>
        </div>
        <div className="bg-gray-800 p-4 rounded">
          <h3 className="text-lg mb-4">Sector Exposure</h3>
          <PieChart width={400} height={300}>
            <Pie
              data={sectorData}
              cx={200}
              cy={150}
              labelLine={false}
              label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              outerRadius={80}
              fill="#8884d8"
              dataKey="value"
            >
              {sectorData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </div>
      </div>
    </>
  );

  const renderPortfolio = () => (
    <div className="bg-gray-800 p-4 rounded">
      <h3 className="text-lg mb-4">Portfolio Holdings</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-left">
          <thead>
            <tr>
              <th className="pb-2">Symbol</th>
              <th className="pb-2">Shares</th>
              <th className="pb-2">Price (₹)</th>
              <th className="pb-2">Value (₹)</th>
              <th className="pb-2">Weight (%)</th>
            </tr>
          </thead>
          <tbody>
            {holdingData.map((holding, index) => (
              <tr key={index} className="border-t border-gray-700">
                <td className="py-2">{holding.ticker}</td>
                <td className="py-2">{holding.quantity ?? holding.shares}</td>
                <td className="py-2">{holding.price.toFixed(2)}</td>
                <td className="py-2">{holding.market_value.toFixed(2)}</td>
                <td className="py-2">{(holding.weight * 100).toFixed(2)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );

  const renderRisk = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gray-800 p-4 rounded">
          <h3 className="text-lg">Sharpe Ratio</h3>
          <p className="text-2xl">{performance.sharpe_ratio.toFixed(2)}</p>
        </div>
        <div className="bg-gray-800 p-4 rounded">
          <h3 className="text-lg">Max Drawdown</h3>
          <p className="text-2xl">{(risk.max_drawdown * 100).toFixed(2)}%</p>
        </div>
        <div className="bg-gray-800 p-4 rounded">
          <h3 className="text-lg">Volatility</h3>
          <p className="text-2xl">{(risk.volatility * 100).toFixed(2)}%</p>
        </div>
      </div>
      <div className="bg-gray-800 p-4 rounded">
        <h3 className="text-lg mb-4">Sector Exposure</h3>
        <BarChart width={600} height={300} data={sectorData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Bar dataKey="value" fill="#8884d8" />
        </BarChart>
      </div>
    </div>
  );

  const renderAlerts = () => (
    <div className="space-y-6">
      <div className="bg-gray-800 p-4 rounded">
        <h3 className="text-lg mb-4">Alerts</h3>
        <ul className="list-disc list-inside">
          {alerts.map((alert, index) => (
            <li key={index} className="mb-2">{alert}</li>
          ))}
        </ul>
      </div>
      <div className="bg-gray-800 p-4 rounded">
        <h3 className="text-lg mb-4">Macro Risk Factors</h3>
        <ul className="list-disc list-inside">
          {macroRiskFactors.map((factor, index) => (
            <li key={index} className="mb-2">{factor}</li>
          ))}
        </ul>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <header className="bg-gray-800 p-4">
        <h1 className="text-2xl font-bold">AMIS — Asset Management Intelligence System</h1>
        <p>Indian Market Dashboard</p>
      </header>
      <nav className="bg-gray-700 p-2">
        <div className="flex space-x-4">
          <button
            className={`px-4 py-2 rounded ${activeTab === 'overview' ? 'bg-blue-600' : 'bg-gray-600'}`}
            onClick={() => setActiveTab('overview')}
          >
            Overview
          </button>
          <button
            className={`px-4 py-2 rounded ${activeTab === 'portfolio' ? 'bg-blue-600' : 'bg-gray-600'}`}
            onClick={() => setActiveTab('portfolio')}
          >
            Portfolio
          </button>
          <button
            className={`px-4 py-2 rounded ${activeTab === 'risk' ? 'bg-blue-600' : 'bg-gray-600'}`}
            onClick={() => setActiveTab('risk')}
          >
            Risk
          </button>
          <button
            className={`px-4 py-2 rounded ${activeTab === 'alerts' ? 'bg-blue-600' : 'bg-gray-600'}`}
            onClick={() => setActiveTab('alerts')}
          >
            Alerts
          </button>
        </div>
      </nav>
      <main className="p-6">
        {activeTab === 'overview' && renderOverview()}
        {activeTab === 'portfolio' && renderPortfolio()}
        {activeTab === 'risk' && renderRisk()}
        {activeTab === 'alerts' && renderAlerts()}
      </main>
    </div>
  );
}

export default App;

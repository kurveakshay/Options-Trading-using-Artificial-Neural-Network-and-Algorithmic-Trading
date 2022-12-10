const chart = LightweightCharts.createChart(document.body, {
  width: 400,
  height: 300,
});
const lineSeries = chart.addLineSeries();
lineSeries.setData([
  { 'time': "2019-04-11", 'value': 80.01 },
  { 'time': "2019-04-12", 'value': 96.63 },
  { 'time': "2019-04-13", 'value': 76.64 },
  { 'time': "2019-04-14", 'value': 81.89 },
]);




function makeDate(){
  object = {};
  object['time'] = "2019-04-11";
  object['value'] = "80.0";
  return object;
}

lineSeries.addLineSeries(makeDate());

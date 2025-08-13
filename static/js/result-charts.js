// Initialize charts when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Helper function to get sentiment color
    function getSentimentColor(sentiment) {
        if (sentiment < -0.33) return 'rgba(220, 53, 69, 0.7)'; // Negative - red
        if (sentiment < 0.33) return 'rgba(108, 117, 125, 0.7)'; // Neutral - gray
        return 'rgba(40, 167, 69, 0.7)'; // Positive - green
    }

    // Helper function to calculate sentiment by outlet
    function calculateSentimentByOutlet(articles) {
        const outlets = {};
        
        articles.forEach(article => {
            const outlet = article.source.name;
            if (!outlets[outlet]) {
                outlets[outlet] = {
                    count: 0,
                    totalSentiment: 0
                };
            }
            outlets[outlet].count++;
            outlets[outlet].totalSentiment += article.sentiment;
        });
        
        const outletArray = Object.keys(outlets).map(outlet => ({
            outlet: outlet,
            count: outlets[outlet].count,
            avgSentiment: outlets[outlet].totalSentiment / outlets[outlet].count
        }));
        
        return outletArray.sort((a, b) => b.count - a.count);
    }
    
    // Helper function to decode HTML entities
    function decodeHtmlEntities(text) {
        if (typeof text !== 'string') {
            return text;
        }
        
        // Create a temporary element to decode HTML entities
        var element = document.createElement('div');
        element.innerHTML = text;
        
        // Log the before and after for debugging
        console.log("Decoding:", text, "→", element.textContent);
        
        return element.textContent || element.innerText || text;
    }

    // Get data from template
    var articles1 = window.articles1 || [];
    var articles2 = window.articles2 || null;
    var analysis1 = window.analysis1 || {};
    var analysis2 = window.analysis2 || null;
    var query1 = window.query1 || "";
    var query2 = window.query2 || "";
    
    // Create sentiment scatter plot
    if (articles1.length > 0) {
        console.log("DEBUG - Creating sentiment scatter plot");
        console.log("DEBUG - Sample article URL:", articles1[0].url);
        
        // Create a simple trace with direct URL links
        var trace1 = {
            x: articles1.map(a => a.publishedAt),
            y: articles1.map(a => a.sentiment),
            text: articles1.map(a => `<b>${a.title}</b><br>Source: ${a.source.name}<br><a href="${a.url}" target="_blank" style="color: #005e30; font-weight: bold; padding: 5px 0; display: inline-block;">Read article →</a>`),
            mode: 'markers',
            type: 'scatter',
            name: query1,
            marker: {
                color: articles1.map(a => getSentimentColor(a.sentiment)),
                size: 10
            },
            hoverinfo: 'text',
            hoverlabel: {
                bgcolor: 'white',
                bordercolor: '#005e30',
                font: {family: 'Arial', size: 12}
            }
        };
        
        var traces = [trace1];
        
        if (articles2 && articles2.length > 0) {
            var trace2 = {
                x: articles2.map(a => a.publishedAt),
                y: articles2.map(a => a.sentiment),
                text: articles2.map(a => `<b>${a.title}</b><br>Source: ${a.source.name}<br><a href="${a.url}" target="_blank" style="color: #00a651; font-weight: bold; padding: 5px 0; display: inline-block;">Read article →</a>`),
                mode: 'markers',
                type: 'scatter',
                name: query2,
                marker: {
                    color: articles2.map(a => getSentimentColor(a.sentiment)),
                    size: 10,
                    opacity: 0.7
                },
                hoverinfo: 'text',
                hoverlabel: {
                    bgcolor: 'white',
                    bordercolor: '#00a651',
                    font: {family: 'Arial', size: 12}
                }
            };
            
            traces.push(trace2);
        }
        
        var layout = {
            title: '',
            xaxis: {
                title: 'Publication Date',
                tickformat: '%b %d, %Y',
                // Add spikes for better hover experience
                showspikes: true,
                spikethickness: 1,
                spikedash: 'solid',
                spikecolor: '#999',
                spikemode: 'across'
            },
            yaxis: {
                title: 'Sentiment Score',
                range: [-1, 1],
                // Add spikes for better hover experience
                showspikes: true,
                spikethickness: 1,
                spikedash: 'solid',
                spikecolor: '#999',
                spikemode: 'across'
            },
            hovermode: 'closest',
            hoverdistance: 100, // Increase hover distance to make it easier to hover
            showlegend: true,
            legend: {
                x: 0,
                y: 1.1,
                orientation: 'h'
            },
            margin: {
                l: 50,
                r: 20,
                t: 10,
                b: 50
            },
            autosize: true
        };
        
        // Configure Plotly with persistent hover mode to allow clicking links
        var config = {
            responsive: true,
            displayModeBar: false,
            // Make hover info persistent until clicked elsewhere
            modeBarButtonsToRemove: ['toImage', 'zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d']
        };
        
        Plotly.newPlot('sentimentScatter', traces, layout, config);
        
        // Add event listener to make hover info persistent when clicked
        var scatterPlot = document.getElementById('sentimentScatter');
        
        // Make hover persistent on click
        scatterPlot.on('plotly_click', function(data) {
            // Keep the hover label visible after clicking
            Plotly.Fx.hover('sentimentScatter', [data.points[0]]);
            
            // Get the URL from the point's data
            var pointIndex = data.points[0].pointIndex;
            var traceIndex = data.points[0].curveNumber;
            var article = traceIndex === 0 ? articles1[pointIndex] : articles2[pointIndex];
            
            // Open the article URL in a new tab
            if (article && article.url) {
                window.open(article.url, '_blank');
            }
        });
    }
    
    // Create timeline chart
    if (analysis1.timeline && analysis1.timeline.length > 0) {
        var timelineCtx = document.getElementById('timelineChart').getContext('2d');
        
        var datasets = [{
            label: query1,
            data: analysis1.timeline.map(item => ({
                x: item.date,
                y: item.count
            })),
            borderColor: '#005e30',
            backgroundColor: 'rgba(0, 94, 48, 0.1)',
            borderWidth: 2,
            tension: 0.1,
            fill: true
        }];
        
        if (analysis2 && analysis2.timeline && analysis2.timeline.length > 0) {
            datasets.push({
                label: query2,
                data: analysis2.timeline.map(item => ({
                    x: item.date,
                    y: item.count
                })),
                borderColor: '#00a651',
                backgroundColor: 'rgba(0, 166, 81, 0.1)',
                borderWidth: 2,
                tension: 0.1,
                fill: true
            });
        }
        
        new Chart(timelineCtx, {
            type: 'line',
            data: {
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'day',
                            displayFormats: {
                                day: 'MMM d'
                            }
                        },
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Number of Articles'
                        },
                        ticks: {
                            precision: 0
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            title: function(context) {
                                const date = new Date(context[0].parsed.x);
                                return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
                            }
                        }
                    }
                }
            }
        });
    }
    
    // Create sources chart
    if (analysis1.sources && analysis1.sources.length > 0) {
        var sourcesCtx = document.getElementById('sourcesChart').getContext('2d');
        
        var topSources1 = analysis1.sources.slice(0, 10);
        var datasets = [{
            label: query1,
            data: topSources1.map(s => s.count),
            backgroundColor: 'rgba(0, 94, 48, 0.7)',
            borderColor: 'rgba(0, 94, 48, 1)',
            borderWidth: 1
        }];
        
        // Debug logging for source names
        console.log("DEBUG - Top sources for query1:", topSources1.map(s => s.name));
        
        // Decode HTML entities in source names
        var labels = topSources1.map(s => decodeHtmlEntities(s.name));
        
        if (analysis2 && analysis2.sources && analysis2.sources.length > 0) {
            var topSources2 = analysis2.sources.slice(0, 10);
            
            // Find unique sources across both queries
            var allSources = new Set([...topSources1.map(s => decodeHtmlEntities(s.name)), ...topSources2.map(s => decodeHtmlEntities(s.name))]);
            labels = Array.from(allSources);
            
            // Create data arrays with 0 for missing sources
            var data1 = labels.map(label => {
                const source = topSources1.find(s => decodeHtmlEntities(s.name) === label);
                return source ? source.count : 0;
            });
            
            var data2 = labels.map(label => {
                const source = topSources2.find(s => decodeHtmlEntities(s.name) === label);
                return source ? source.count : 0;
            });
            
            datasets = [
                {
                    label: query1,
                    data: data1,
                    backgroundColor: 'rgba(0, 94, 48, 0.7)',
                    borderColor: 'rgba(0, 94, 48, 1)',
                    borderWidth: 1
                },
                {
                    label: query2,
                    data: data2,
                    backgroundColor: 'rgba(0, 166, 81, 0.7)',
                    borderColor: 'rgba(0, 166, 81, 1)',
                    borderWidth: 1
                }
            ];
        }
        
        new Chart(sourcesCtx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                scales: {
                    x: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Number of Articles'
                        },
                        ticks: {
                            precision: 0
                        }
                    },
                    y: {
                        ticks: {
                            // Force decoding of HTML entities in labels
                            callback: function(value, index) {
                                return decodeHtmlEntities(this.getLabelForValue(index));
                            }
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            // Decode HTML entities in tooltips
                            title: function(tooltipItems) {
                                return decodeHtmlEntities(tooltipItems[0].label);
                            }
                        }
                    }
                }
            }
        });
    }
    
    // Create sentiment distribution pie charts
    if (articles1.length > 0) {
        var sentimentPieCtx1 = document.getElementById('sentimentPieChart1').getContext('2d');
        
        var positive1 = articles1.filter(a => a.sentiment > 0.2).length;
        var neutral1 = articles1.filter(a => a.sentiment >= -0.2 && a.sentiment <= 0.2).length;
        var negative1 = articles1.filter(a => a.sentiment < -0.2).length;
        
        new Chart(sentimentPieCtx1, {
            type: 'pie',
            data: {
                labels: ['Positive', 'Neutral', 'Negative'],
                datasets: [{
                    data: [positive1, neutral1, negative1],
                    backgroundColor: [
                        'rgba(40, 167, 69, 0.7)',
                        'rgba(108, 117, 125, 0.7)',
                        'rgba(220, 53, 69, 0.7)'
                    ],
                    borderColor: [
                        'rgba(40, 167, 69, 1)',
                        'rgba(108, 117, 125, 1)',
                        'rgba(220, 53, 69, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const value = context.raw;
                                const percentage = Math.round((value / articles1.length) * 100);
                                return `${context.label}: ${value} (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        });
        
        if (articles2 && articles2.length > 0) {
            var sentimentPieCtx2 = document.getElementById('sentimentPieChart2').getContext('2d');
            
            var positive2 = articles2.filter(a => a.sentiment > 0.2).length;
            var neutral2 = articles2.filter(a => a.sentiment >= -0.2 && a.sentiment <= 0.2).length;
            var negative2 = articles2.filter(a => a.sentiment < -0.2).length;
            
            new Chart(sentimentPieCtx2, {
                type: 'pie',
                data: {
                    labels: ['Positive', 'Neutral', 'Negative'],
                    datasets: [{
                        data: [positive2, neutral2, negative2],
                        backgroundColor: [
                            'rgba(40, 167, 69, 0.7)',
                            'rgba(108, 117, 125, 0.7)',
                            'rgba(220, 53, 69, 0.7)'
                        ],
                        borderColor: [
                            'rgba(40, 167, 69, 1)',
                            'rgba(108, 117, 125, 1)',
                            'rgba(220, 53, 69, 1)'
                        ],
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const value = context.raw;
                                    const percentage = Math.round((value / articles2.length) * 100);
                                    return `${context.label}: ${value} (${percentage}%)`;
                                }
                            }
                        }
                    }
                }
            });
        }
    }
    
    // Create sentiment by outlet charts
    if (articles1.length > 0) {
        var sentimentByOutlet1 = calculateSentimentByOutlet(articles1);
        var topOutlets1 = sentimentByOutlet1.slice(0, 15);
        
        if (topOutlets1 && topOutlets1.length > 0) {
            new Chart(document.getElementById('sentimentByOutletChart1'), {
                type: 'bar',
                data: {
                    labels: topOutlets1.map(item => `${decodeHtmlEntities(item.outlet)} (${item.count})`),
                    datasets: [{
                        label: 'Average Sentiment',
                        data: topOutlets1.map(item => item.avgSentiment),
                        backgroundColor: topOutlets1.map(item => getSentimentColor(item.avgSentiment)),
                        borderColor: 'rgba(0, 0, 0, 0.1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'y',
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const sentiment = context.raw;
                                    let label = `Sentiment: ${sentiment.toFixed(2)}`;
                                    if (sentiment < -0.33) {
                                        label += ' (Negative)';
                                    } else if (sentiment < 0.33) {
                                        label += ' (Neutral)';
                                    } else {
                                        label += ' (Positive)';
                                    }
                                    return label;
                                }
                            }
                        }
                    },
                    scales: {
                        y: {
                            ticks: {
                                autoSkip: false,
                                maxRotation: 0,
                                padding: 10
                            }
                        },
                        x: {
                            beginAtZero: false,
                            min: -1,
                            max: 1,
                            ticks: {
                                callback: function(value) {
                                    if (value === -1) return 'Negative';
                                    if (value === 0) return 'Neutral';
                                    if (value === 1) return 'Positive';
                                    return '';
                                }
                            }
                        }
                    }
                }
            });
        }
        
        // Create sentiment by outlet chart for query2 if it exists
        if (articles2 && articles2.length > 0) {
            var sentimentByOutlet2 = calculateSentimentByOutlet(articles2);
            var topOutlets2 = sentimentByOutlet2.slice(0, 15);
            
            if (topOutlets2 && topOutlets2.length > 0) {
                new Chart(document.getElementById('sentimentByOutletChart2'), {
                    type: 'bar',
                    data: {
                        labels: topOutlets2.map(item => `${decodeHtmlEntities(item.outlet)} (${item.count})`),
                        datasets: [{
                            label: 'Average Sentiment',
                            data: topOutlets2.map(item => item.avgSentiment),
                            backgroundColor: topOutlets2.map(item => getSentimentColor(item.avgSentiment)),
                            borderColor: 'rgba(0, 0, 0, 0.1)',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        indexAxis: 'y',
                        plugins: {
                            legend: {
                                display: false
                            },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        const sentiment = context.raw;
                                        let label = `Sentiment: ${sentiment.toFixed(2)}`;
                                        if (sentiment < -0.33) {
                                            label += ' (Negative)';
                                        } else if (sentiment < 0.33) {
                                            label += ' (Neutral)';
                                        } else {
                                            label += ' (Positive)';
                                        }
                                        return label;
                                    }
                                }
                            }
                        },
                        scales: {
                            y: {
                                ticks: {
                                    autoSkip: false,
                                    maxRotation: 0,
                                    padding: 10
                                }
                            },
                            x: {
                                beginAtZero: false,
                                min: -1,
                                max: 1,
                                ticks: {
                                    callback: function(value) {
                                        if (value === -1) return 'Negative';
                                        if (value === 0) return 'Neutral';
                                        if (value === 1) return 'Positive';
                                        return '';
                                    }
                                }
                            }
                        }
                    }
                });
            }
        }
    }
    
    // Copy to Claude button functionality
    document.getElementById('copyToClaudeBtn').addEventListener('click', function() {
        var content = '';
        
        // Add query info
        content += `Media Analysis for "${query1}"`;
        if (query2) content += ` vs "${query2}"`;
        content += '\n\n';
        
        // Add date range
        if (analysis1 && analysis1.date_range) {
            content += `Date Range: ${analysis1.date_range.start} to ${analysis1.date_range.end}\n\n`;
        }
        
        // Add coverage metrics
        content += '## Coverage Metrics\n\n';
        content += `${query1}: ${analysis1.total_articles} articles, Avg Sentiment: ${analysis1.avg_sentiment.toFixed(2)}\n`;
        if (analysis2) {
            content += `${query2}: ${analysis2.total_articles} articles, Avg Sentiment: ${analysis2.avg_sentiment.toFixed(2)}\n`;
        }
        content += '\n';
        
        // Add top sources
        content += '## Top Sources\n\n';
        if (analysis1 && analysis1.sources) {
            content += `${query1}: `;
            content += analysis1.sources.slice(0, 5).map(s => `${s.name} (${s.count})`).join(', ');
            content += '\n';
        }
        if (analysis2 && analysis2.sources) {
            content += `${query2}: `;
            content += analysis2.sources.slice(0, 5).map(s => `${s.name} (${s.count})`).join(', ');
            content += '\n';
        }
        content += '\n';
        
        // Add top topics
        content += '## Top Topics\n\n';
        if (analysis1 && analysis1.topics) {
            content += `${query1}: `;
            content += analysis1.topics.slice(0, 10).map(t => `${t.topic} (${t.count})`).join(', ');
            content += '\n';
        }
        if (analysis2 && analysis2.topics) {
            content += `${query2}: `;
            content += analysis2.topics.slice(0, 10).map(t => `${t.topic} (${t.count})`).join(', ');
            content += '\n';
        }
        content += '\n';
        
        // Add articles
        content += '## Articles\n\n';
        
        // Add articles for query1
        content += `### ${query1} Articles\n\n`;
        articles1.slice(0, 10).forEach(article => {
            content += `- **${article.title}** (${article.source.name})\n`;
            content += `  Published: ${new Date(article.publishedAt).toLocaleDateString()}\n`;
            content += `  Sentiment: ${article.sentiment.toFixed(2)}\n`;
            content += `  URL: ${article.url}\n\n`;
        });
        
        // Add articles for query2 if it exists
        if (articles2 && articles2.length > 0) {
            content += `### ${query2} Articles\n\n`;
            articles2.slice(0, 10).forEach(article => {
                content += `- **${article.title}** (${article.source.name})\n`;
                content += `  Published: ${new Date(article.publishedAt).toLocaleDateString()}\n`;
                content += `  Sentiment: ${article.sentiment.toFixed(2)}\n`;
                content += `  URL: ${article.url}\n\n`;
            });
        }
        
        // Copy to clipboard
        navigator.clipboard.writeText(content).then(function() {
            alert('Analysis copied to clipboard! You can now paste it into Claude.');
        }, function(err) {
            console.error('Could not copy text: ', err);
            alert('Failed to copy to clipboard. Please try again.');
        });
    });
    
    // Share button functionality
    document.getElementById('shareBtn').addEventListener('click', function() {
        // Get current URL
        var url = window.location.href;
        
        // Copy to clipboard
        navigator.clipboard.writeText(url).then(function() {
            alert('URL copied to clipboard! You can now share it with others.');
        }, function(err) {
            console.error('Could not copy text: ', err);
            alert('Failed to copy to clipboard. Please try again.');
        });
    });
});

import plotly.express as px

from plotly.offline import plot
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from hydroanalytics.precipitation.precipitation import *
from hydroanalytics.precipitation.tests.data import CVG_PRECIP

HERE = os.path.dirname(os.path.realpath(__file__))


class TestEvent(unittest.TestCase):
    def setUp(self) -> None:
        self.rainfall_data = pd.read_csv(
            CVG_PRECIP,
            index_col=0,
            parse_dates=True,
        )

        self.rainfall_data = self.rainfall_data[self.rainfall_data.index.year >= 2004]

    def test_get_events(self) -> None:
        """
        This function tests the get_events function
        :return:
        """
        events = get_events(self.rainfall_data, floor=0.0)
        events = events.loc[(events.precip_total > 0.05) & (events.duration > pd.Timedelta('1h'))]
        events_cp = get_event_attributes(
            latitude=39.0470,
            longitude=-84.6656,
            events=events,
        )

        self.events = events_cp
        self.events['duration_hours'] = self.events['duration'] / pd.Timedelta('1h')

        clustered_events = cluster_events(
            events=self.events,
            cluster_columns=['duration_hours', 'precip_total', 'precip_peak', 'return_period'],
            number_of_clusters=15,
        )

        clustered_events['cluster'] = clustered_events['cluster'].astype('category')

        for cluster, cluster_group in clustered_events.groupby('cluster'):
            clustered_events.loc[cluster_group.index, 'cluster_size'] = cluster_group.shape[0]
            clustered_events.loc[cluster_group.index, 'cluster_average_duration'] = cluster_group['duration_hours'].mean()
            clustered_events.loc[cluster_group.index, 'cluster_average_precip_total'] = cluster_group['precip_total'].mean()
            clustered_events.loc[cluster_group.index, 'cluster_average_precip_peak'] = cluster_group['precip_peak'].mean()
            clustered_events.loc[cluster_group.index, 'cluster_average_return_period'] = cluster_group['return_period'].mean()
            sample = cluster_group.sample(1)
            clustered_events.loc[cluster_group.index, 'cluster_sample'] = False
            clustered_events.loc[sample.index, 'cluster_sample'] = True

        clustered_events.loc[clustered_events['cluster_sample'] == True, 'marker_size'] = 3
        clustered_events.loc[clustered_events['cluster_sample'] == False, 'marker_size'] = 0.5

        fig = px.scatter_matrix(
            clustered_events,
            dimensions=['duration_hours', 'precip_total', 'precip_peak', 'return_period'],
            color="cluster",
            symbol="cluster_sample",
            size=clustered_events['marker_size'],
            hover_data= [
                'name',
                'start',
                'end',
                'actual_end',
                'duration',
                'post_event_time',
                'antecedent_event_time',
                'precip_peak',
                'precip_total',
                'return_period',
                'duration_hours',
                'cluster',
                'cluster_size',
                'cluster_average_duration',
                'cluster_average_precip_total',
                'cluster_average_precip_peak',
                'cluster_average_return_period',
                'cluster_sample'
            ],
            labels={
                'cluster': 'Cluster',
                'cluster_sample': 'Cluster Sample',
                'duration_hours': 'Duration (hours)',
                'precip_total': 'Precipitation (in)',
                'precip_peak': 'Peak Intensity (in/5 min)',
                'return_period': 'Return Period (years)',
            },
            category_orders={
                'cluster': list(range(15)),
            },
        )

        plot(fig, filename='scatter_matrix.html')
        clustered_events.to_csv('clustered_events.csv')

        sub_plot_fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            x_title='Date and Time',
            row_titles=['Events', 'Precipitation (in)'],
            row_heights=[0.2, 0.8],
        )

        clustered_events['ylabel'] = 1.0
        fig_timeline = px.timeline(
            clustered_events,
            x_start="start",
            x_end="end",
            y="ylabel",
            color="cluster",
            hover_data=['name', 'start', 'end', 'duration_hours', 'precip_total', 'precip_peak', 'return_period'],
            labels={
                'duration_hours': 'Duration (hours)',
                'precip_total': 'Precipitation (in)',
                'precip_peak': 'Peak Intensity (in/5 min)',
                'return_period': 'Return Period (years)',
            },
            category_orders={
                'cluster': list(range(16)),
            },
        )

        for bar in fig_timeline.data:
            # bar['orientation'] = 'v'
            sub_plot_fig.add_trace(
                bar,
                row=1,
                col=1,
            )

        hourly_resample = self.rainfall_data[['precip']].resample('H').sum()
        # sub_plot_fig.add_trace(
        #     go.Scatter(
        #         x=hourly_resample.index,
        #         y=hourly_resample['precip'],
        #         fill='tozeroy',
        #         name='Precipitation (in)',
        #     ),
        #     row=2,
        #     col=1,
        # )

        sub_plot_fig.add_trace(
            go.Scatter(
                x= self.rainfall_data.index,
                y= self.rainfall_data['precip'],
                fill='tozeroy',
                name='Precipitation (in)',
            ),
            row=2,
            col=1,
        )

        sub_plot_fig.update_layout(fig_timeline.layout)
        plot(figure_or_data=sub_plot_fig, filename='cluster_bar.html')

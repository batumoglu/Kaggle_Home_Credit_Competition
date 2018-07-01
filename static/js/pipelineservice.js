/*!
 * PipelineService v1.0.0
 * Author: Ugur Uker
 * Released under the MIT License.
 */

var socket = io.connect('http://' + document.domain + ':' + location.port);

socket.on('itemadded', function(item) {
    $('#' + item + '-state').removeClass('loader')
    $('#' + item + '-state').addClass('checkmark')
});

socket.on('itemremoved', function(item) {
    $('#' + item + '-state').removeClass('checkmark')
    $('#' + item + '-state').removeClass('loader')
});

socket.on('itemremoved', function(item) {
    $('#' + item + '-state').removeClass('checkmark')
    $('#' + item + '-state').removeClass('loader')
});

socket.on('schedulechanged', function(sch) {
    var schedule = JSON.parse(sch)
    $('#joblist').empty()
    if(schedule.jobs.length > 0) {
        var jobs = new JobList().compile(schedule);
        $('#joblist').append(jobs);
    }
});

function JobList(){
    row_template = `
    <tr id="job_id">
        <th scope="row">job_index</th>
        <td>2018-07-01 14:43:23.999</td>
        <td>model_id</td>
        <td>dataset_id</td>
        <td><a href="http://localhost:5000/score/job_id" target="_blank">0.7976</a></td>
        <td>@mdo</td>
    </tr>`;
    
    this.compile = function(schedule){
        joblist_str  = ''
        schedule.jobs.forEach(function(job){
            joblist_str += row_template
            .replaceAll('job_id',job[1] + '-' + job[2])
            .replaceAll('job_index',job[0])
            .replaceAll('model_id',job[1])
            .replaceAll('dataset_id',job[2])
        });
        return joblist_str
    }
};

String.prototype.replaceAll = function(search, replacement) {
    var target = this;
    return target.replace(new RegExp(search, 'g'), replacement);
};
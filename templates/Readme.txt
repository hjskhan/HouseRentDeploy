<span class="form-label" for="OverallQual">Overall Quality</span>
<select class="form-control" id="OverallQual" name="OverallQual"


<div class="form-group">
									<span class="form-label" for="YearBuilt">Year Built</span>
									<input class="form-control" type="number" name="YearBuilt" id="datepicker"
										placeholder="Select Year" required>
									<span class="select-arrow"></span>
									<!-- Bootstrap JS -->
									<script src="https://code.jquery.com/jquery-3.3.1.slim.min.js"></script>
									<script
										src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js"></script>
									<script
										src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js"></script>
									<!-- Bootstrap Datepicker JS -->
									<script
										src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap-datepicker/1.9.0/js/bootstrap-datepicker.min.js"></script>

									<script>
										$(document).ready(function () {
											$('#datepicker').datepicker({
												format: "yyyy",
												viewMode: "years",
												minViewMode: "years",
												startDate: '1872',
												endDate: '2010'
											});
										});
									</script>
								</div>
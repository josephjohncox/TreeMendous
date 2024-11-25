use pyo3::prelude::*;
use std::collections::BTreeMap;

#[pyclass]
struct IntervalManager {
    intervals: BTreeMap<i32, i32>,
    total_available_length: i32,
}

#[pymethods]
impl IntervalManager {
    #[new]
    fn new() -> Self {
        IntervalManager {
            intervals: BTreeMap::new(),
            total_available_length: 0,
        }
    }

    fn release_interval(&mut self, start: i32, end: i32) {
        if start >= end {
            return;
        }

        let mut start = start;
        let mut end = end;
        let mut intervals_to_remove = vec![];

        // Merge with previous intervals if overlapping
        if let Some((&prev_start, &prev_end)) = self.intervals.range(..=start).next_back() {
            if prev_end >= start {
                start = prev_start;
                end = end.max(prev_end);
                self.total_available_length -= prev_end - prev_start;
                intervals_to_remove.push(prev_start);
            }
        }

        // Merge with subsequent overlapping intervals
        for (&curr_start, &curr_end) in self.intervals.range(start..).take_while(|(&s, _)| s <= end).collect::<Vec<_>>() {
            end = end.max(curr_end);
            self.total_available_length -= curr_end - curr_start;
            intervals_to_remove.push(curr_start);
        }

        // Remove merged intervals
        for key in intervals_to_remove {
            self.intervals.remove(&key);
        }

        // Insert the new merged interval
        self.intervals.insert(start, end);
        self.total_available_length += end - start;
    }

    fn reserve_interval(&mut self, start: i32, end: i32) {
        if start >= end {
            return;
        }

        let mut intervals_to_remove = vec![];
        let mut intervals_to_add = vec![];

        // Find all overlapping intervals
        for (&curr_start, &curr_end) in self.intervals.range(..).collect::<Vec<_>>() {
            if curr_end <= start {
                continue;
            }
            if curr_start >= end {
                break;
            }

            // Overlapping interval found
            intervals_to_remove.push(curr_start);
            self.total_available_length -= curr_end - curr_start;

            // Add non-overlapping parts back
            if curr_start < start {
                intervals_to_add.push((curr_start, start));
            }
            if curr_end > end {
                intervals_to_add.push((end, curr_end));
            }
        }

        // Remove overlapping intervals
        for key in intervals_to_remove {
            self.intervals.remove(&key);
        }

        // Add back non-overlapping intervals
        for (s, e) in intervals_to_add {
            self.intervals.insert(s, e);
            self.total_available_length += e - s;
        }
    }

    fn find_interval(&self, point: i32, length: i32) -> Option<(i32, i32)> {
        // Check for intervals containing 'point'
        if let Some((&start, &end)) = self.intervals.range(..=point).next_back() {
            if end - point >= length {
                return Some((point, point + length));
            }
        }

        // Check for intervals starting after 'point'
        for (&start, &end) in self.intervals.range(point..).take(1) {
            if end - start >= length {
                return Some((start, start + length));
            }
        }

        None
    }

    fn get_total_available_length(&self) -> i32 {
        self.total_available_length
    }

    fn print_intervals(&self) {
        println!("Available intervals:");
        for (&s, &e) in &self.intervals {
            println!("[{}, {})", s, e);
        }
        println!("Total available length: {}", self.total_available_length);
    }

    fn get_intervals(&self) -> Vec<(i32, i32)> {
        self.intervals.iter().map(|(&s, &e)| (s, e)).collect()
    }
}

#[pymodule]
fn interval_manager(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<IntervalManager>()?;
    Ok(())
}
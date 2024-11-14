import React, { useState } from 'react';
import { Modal, Box, Typography, Button } from '@mui/material';

const Popup: React.FC<{ text: string }> = ({ text }) => {
  const [open, setOpen] = useState(false);

  const handleOpen = () => setOpen(true);
  const handleClose = () => setOpen(false);

  return (
    <div>
      <Button variant="text" color="primary" onClick={handleOpen}>
        {text}
      </Button>
      <Modal
        open={open}
        onClose={handleClose}
        aria-labelledby="popup-title"
        aria-describedby="popup-description"
      >
        <Box
          sx={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            width: 300,
            bgcolor: 'background.paper',
            borderRadius: 2,
            boxShadow: 24,
            p: 4,
          }}
        >
          <Typography id="popup-description" variant="body1">
            גרסה: 0.1.0
            פותח ע"י מטריקס ישראל בשיתוף עם התוכנית הישראלית ל-NLP
          </Typography>
          <Box mt={2} textAlign="center">
            <Button variant="outlined" color="secondary" onClick={handleClose}>
              סגור
            </Button>
          </Box>
        </Box>
      </Modal>
    </div>
  );
};

export default Popup;
